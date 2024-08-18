# Ateema AI Video Generation

## Overview
This project is focused on generating videos that synchronize voiceovers with images and animations using various AI-driven tools such as Stable Diffusion, Tortoise TTS, SadTalker, and MoviePy. The goal is to create visually appealing and engaging videos based on user-provided text recommendations.

## Features
- **Text-to-Speech (TTS)**: Converts text into speech using Tortoise TTS.
- **Image Generation**: Creates or downloads images related to the text using Stable Diffusion.
- **Video Synchronization**: Synchronizes images and speech to produce a cohesive video using MoviePy.
- **Avatar Creation with SadTalker**: Generates an animated avatar that lip-syncs with the audio.
- **Google Drive Integration**: Automatically uploads the final video to Google Drive for easy sharing.
- **Generating Video Triggering App**: A separate app that sends requests to the control system to trigger the video generation process.
- **Manual Video Generation Approach**: Allows for manual creation and processing of videos, offering an alternative to the automated approach.

## Setup
### Environment Requirements
- Python 3.10 or later
- Jupyter Notebook or Google Colab environment

### Dependencies
The project requires several Python packages, including:
- `torch`
- `sadtalker`
- `diffusers`
- `requests`
- `moviepy`
- `Pillow`
- `google-auth`
- `google-api-python-client`
- `pyngrok`

### Cloning Repositories
Specific repositories are required for TTS and avatar creation, which need to be cloned and set up.

## Usage
### Text-to-Speech Conversion
The provided text recommendations are processed into speech. The text is cleaned, split into manageable chunks, and then converted into audio.

### Image Generation and Processing
Images are either generated using Stable Diffusion or downloaded based on provided URLs. These images are labeled and prepared for synchronization in the video.

### Avatar Creation with SadTalker
SadTalker is used to create an animated avatar that synchronizes with the generated audio, adding a visual element to the voiceover.

### Video Generation
The video is created by synchronizing the generated images and the avatar with the voiceover. MoviePy is used to compile these elements into the final video.

### Uploading to Google Drive
The completed video is automatically uploaded to Google Drive for easy access and sharing.

### Generating Video Triggering App
The project includes an app that sends requests to the control system, triggering the video generation process. This app allows for seamless automation and control of the video creation workflow.

### Manual Video Generation Approach
In addition to the automated process, the project supports a manual approach to video generation. This includes manually generating the audio, images, and video components, providing flexibility for specific use cases.

## Troubleshooting
### Common Issues
- Ensure all dependencies are installed.
- Verify that paths to files (e.g., images, audio, models) are correct.
- Check the SadTalker setup if the avatar isn't generated correctly.
- Re-authenticate Google Drive API access if uploading fails.
