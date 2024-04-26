# Web Scraping

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to scrape image URLs and count <p> tags within a specific section
def scrape_images_and_paragraphs(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the specific section with <article> tag and class "post py-3 pt-md-4 px-1 px-md-5 card-body"
        section = soup.find('article', class_='post py-3 pt-md-4 px-1 px-md-5 card-body')

        # Find all <img> tags within the section
        images = section.find_all('img')

        # Create lists to store image URLs and paragraph counts
        img_urls = []
        p_counts = []

        # Loop through each image
        for img in images:
            # Check if the 'src' attribute exists
            if 'src' in img.attrs:
                # Get the image URL
                img_url = img['src']
                img_urls.append(img_url)

                # Count <p> tags before the image within the section
                p_count = len(img.find_all_previous('p'))
                p_counts.append(p_count)

        return img_urls, p_counts
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None, None

# Function to scrape titles, descriptions, and paragraph indexes
def scrape_titles_descriptions_and_indexes(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all paragraphs
        paragraphs = soup.find_all('p')

        # Find all titles (text within <strong> tags) and descriptions
        data = []
        for idx, p in enumerate(paragraphs):
            strong_tags = p.find_all('strong')
            if strong_tags:
                title = strong_tags[0].get_text()
                description = p.get_text().replace(title, '').strip()
                data.append({'title': title, 'description': description, 'Paragraph Index': idx})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/families/30-kid-friendly-things-to-do-in-chicago/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')

# Drop unnecessary columns
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image'], inplace=True)

# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_family.csv', index=False)

print("Combined CSV file created successfully.")

# Data Preprocessing

import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_family.csv', skiprows=[1])
# print(df.columns)

punctuation = '!”$%&\’()*+,-./:;<=>?[\\]^_`{|}~•@©'

def remove_links(text):
    """Takes a string and removes web links from it"""
    text = re.sub(r'http\S+', '', text)  # remove http links
    text = re.sub(r'bit.ly/\S+', '', text)  # remove bitly links
    text = text.strip('[link]')  # remove [links]
    text = text.strip()
    text = re.sub(r'pic.twitter\S+', '', text)
    return text


def remove_tags(text):
    remove = re.compile(r'')
    return re.sub(remove, '', text)


def remove_users(text):
    """Takes a string and removes retweet and @user information"""
    text = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', text)  # remove re-tweet
    text = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', text)  # remove tweeted at
    return text


def remove_hashtags(text):
    """Takes a string and removes any hash tags"""
    text = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', text)  # remove hash tags
    return text


def remove_av(text):
    """Takes a string and removes AUDIO/VIDEO tags or labels"""
    text = re.sub('VIDEO:', '', text)  # remove 'VIDEO:' from start of tweet
    text = re.sub('AUDIO:', '', text)  # remove 'AUDIO:' from start of tweet
    return text


def basic_clean(text):
    """Main master function to clean tweets only without tokenization or removal of stopwords"""
    text = remove_users(text)
    text = remove_links(text)
    text = remove_hashtags(text)
    text = remove_tags(text)
    text = remove_av(text)
    text = text.lower()  # lower case
    text = re.sub('[' + punctuation + ']+', ' ', text)  # strip punctuation
    text = re.sub('\s+', ' ', text)  # remove double spacing
    text = re.sub('([0-9]+)', '', text)  # remove numbers
    text = re.sub('📝 …', '', text)
    text = re.sub('–', '', text)
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [x for x in words if x not in stop_words]


def lemmatize_word(text):
    wordnet = WordNetLemmatizer()
    return " ".join([wordnet.lemmatize(word) for word in text])

# Apply basic cleaning to the 'description' column and store the result in 'text_cleaned'
df['text_cleaned'] = df['description'].apply(basic_clean)
# df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords)
# df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_word)

df.to_csv('stage1_data_family.csv', index=False)



