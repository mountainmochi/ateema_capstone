import re
import torch
import torchaudio
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import pandas as pd
from diffusers import StableDiffusionPipeline
import moviepy.editor as mp
import shutil
import logging
from flask import Flask, request, jsonify
import json
from threading import Thread
import uuid
from datetime import datetime
from google.colab import userdata
from pyngrok import ngrok

class RecommendationProcessor:
    def __init__(self):
        self.logger = self.setup_logger()
        self.model = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
    
    def setup_logger(self):
        logger = logging.getLogger('RecommendationProcessor')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def dynamically_modify_recommendation(self, text):
        pattern_remove = r"(AI tour guide\.\s)(.*?)(I'd be happy to recommend some interesting places to visit\.\s)"
        modified_text = re.sub(pattern_remove, r"\1\3", text)
        return modified_text

    def replace_numbers_with_natural_transitions(self, text):
        transitions = [
            "First up,", "Following that,", "Another great option is,", "You might also like,",
            "For something different,", "If you're looking for more,", "Don't miss,", "A must-visit is,", "Lastly,"
        ]
        pattern = r'\b(\d+)\.\s'
        transition_index = 0

        def replace_match(match):
            nonlocal transition_index
            transition = transitions[transition_index % len(transitions)]
            transition_index += 1
            return f"{transition} {match.group(0)[3:]}"  # Skip the "1. ", "2. ", etc.

        result = re.sub(pattern, replace_match, text)
        return result

    def clean_text(self, text):
        self.logger.info("Cleaning text")
        cleaned = re.sub(r'\*+', '', text)
        return cleaned.strip()

    def process_in_chunks(self, text, chunk_size=500):
        self.logger.info("Processing text in chunks")
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            if len(' '.join(current_chunk)) + len(sentence) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
            current_chunk.append(sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def generate_audio(self, recommendation):
        self.logger.info("Generating audio")
        clean_recommendation = self.clean_text(recommendation)
        chunks = self.process_in_chunks(clean_recommendation)
        self.logger.info(f"Number of chunks: {len(chunks)}")

        preset = "ultra_fast"
        voice = 'emma'
        voice_samples, conditioning_latents = load_voice(voice)

        all_audio = []

        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            audio = self.tts.tts_with_preset(chunk, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)
            all_audio.append(audio.cpu().squeeze(0))

        combined_audio = torch.cat([a.view(1, -1) for a in all_audio], dim=1)
        os.chdir('/content')
        torchaudio.save('generated_combined.wav', combined_audio, 24000)
        self.logger.info(f"Total audio length: {combined_audio.shape[1]} frames")
        return '/content/generated_combined.wav'

    def generate_or_download_image_with_label(self, entry):
        self.logger.info(f"Generating or downloading image for: {entry['Title']}")
        place = entry["Title"]

        if entry["Match"] == "Yes" and entry["URL"] != "No URL available":
            response = requests.get(entry["URL"])
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            with torch.no_grad():
                image = self.model(f"{place} is either a restaurant or event in Chicago").images[0]
            image = image.convert("RGB")

        draw = ImageDraw.Draw(image)

        try:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            font_size = 80
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
            font_size = 40

        max_width = image.width - 20
        text_bbox = draw.textbbox((0, 0), place, font=font)
        text_width = text_bbox[2] - text_bbox[0]

        while text_width > max_width and font_size > 10:
            font_size -= 5
            font = ImageFont.truetype(font_path, font_size)
            text_bbox = draw.textbbox((0, 0), place, font=font)
            text_width = text_bbox[2] - text_bbox[0]

        text_height = text_bbox[3] - text_bbox[1]
        text_position = ((image.width - text_width) / 2, 20)

        rectangle_padding = 10
        rect_position = [
            text_position[0] - rectangle_padding,
            text_position[1] - rectangle_padding,
            text_position[0] + text_width + rectangle_padding,
            text_position[1] + text_height + rectangle_padding
        ]
        draw.rectangle(rect_position, fill="black")

        draw.text(text_position, place, font=font, fill="white")

        return image

    def process_all_images(self, url_information):
        self.logger.info("Processing all images")
        if not os.path.exists('video_image'):
            os.makedirs('video_image')

        for entry in url_information:
            image = self.generate_or_download_image_with_label(entry)
            filename = entry["Title"].replace(" ", "_")
            image.save(f"video_image/{filename}.png")

    def trigger_video_generation(self, text_recommendation, url_information):
        self.logger.info("Triggering video generation process")

        # Step 1: Modify the text recommendation and replace transitions
        modified_text = self.dynamically_modify_recommendation(text_recommendation)
        final_text = self.replace_numbers_with_natural_transitions(modified_text)

        # Step 2: Generate audio from the text recommendation
        audio_path = self.generate_audio(final_text)

        # Step 3: Process images
        self.process_all_images(url_information)

        # Step 4: Generate the final video
        video_path = self.generate_video(image_folder="video_image", audio_path=audio_path, avatar_video_path=None)

        self.logger.info("Video generation triggered successfully")
        return video_path

    def upload_to_drive(self, video_path, credentials_path, folder_id=None):
        self.logger.info("Uploading video to Google Drive")
        drive = GoogleDrive(self.authenticate_drive(credentials_path))
        file_metadata = {'name': os.path.basename(video_path)}
        if folder_id:
            file_metadata['parents'] = [folder_id]
        file = drive.CreateFile(file_metadata)
        file.SetContentFile(video_path)
        file.Upload()
        self.logger.info(f"Video uploaded to Google Drive with ID: {file['id']}")
        return file['id']

    def authenticate_drive(self, credentials_path):
        self.logger.info("Authenticating Google Drive")
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile(credentials_path)
        if gauth.credentials is None:
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            gauth.Refresh()
        else:
            gauth.Authorize()
        gauth.SaveCredentialsFile(credentials_path)
        return gauth


# Flask application setup
app = Flask(__name__)
processor = RecommendationProcessor()
jobs = {}

@app.route('/webhook', methods=['POST'])
def typeform_webhook():
    try:
        request_data = request.get_json()
        if request_data is None:
            return jsonify({'message': 'No JSON data received'}), 400

        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # Store initial job status
        jobs[job_id] = {'status': 'processing', 'result': None}

        # Start processing in a background thread
        Thread(target=process_video, args=(job_id, request_data)).start()

        # Immediately return a success response with the job ID
        return jsonify({'message': 'Processing started', 'job_id': job_id}), 202
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'message': 'Error processing request', 'error': str(e)}), 500

def process_video(job_id, input_data):
    try:
        recommendation, image_url = process_input(input_data)
        video_url = processor.trigger_video_generation(recommendation, image_url)

        # Store the result
        jobs[job_id] = {'status': 'completed', 'result': video_url}
        print(f"Video processing completed: {video_url}")
    except Exception as e:
        jobs[job_id] = {'status': 'failed', 'result': str(e)}
        print(f"Error in background processing: {str(e)}")

@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    job = jobs.get(job_id)
    print(job_id)
    if not job:
        return jsonify({'message': 'Job not found'}), 404
    return jsonify(job)

def process_input(input_data):
    # If input_data is already a dictionary, no need to parse it
    if isinstance(input_data, dict):
        data = input_data
    else:
        # If it's a string, parse it as JSON
        try:
            data = json.loads(input_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None, None

    print(data)

    # Extract the recommendation as a string
    recommendation = data['recommendation']

    # Extract the image_url information
    image_url = data['url_information']

    return recommendation, image_url


if __name__ == '__main__':
    # Setup and start Flask app with ngrok
    ngrok.set_auth_token(userdata.get('ngrok_key'))
    ngrok_tunnel = ngrok.connect(5000)
    print('Public URL:', ngrok_tunnel.public_url)
    app.run(port=5000)