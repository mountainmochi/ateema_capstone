#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk_stopwords = stopwords.words('english')


# In[14]:


## 30 free things to do in Chicago

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
                data.append({'title': title, 'description': description,'personas': 'budget', 'Paragraph Index': idx, 'category': 'free things to do'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/tours-and-attractions/free-chicago-museums-attractions/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts, 'category': 'image'})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')

# Drop unnecessary columns (category_y)
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image', 'category_y'], inplace=True)


# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_Freethings.csv', index=False)

print("Combined CSV file created successfully.")


# In[15]:


## 30 free things to do in Chicago cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_Freethings.csv', skiprows=[1])
# Web Scraping

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
                data.append({'title': title, 'description': description, 'personas': 'budget', 'Paragraph Index': idx, 'category': 'free things to do'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/tours-and-attractions/free-chicago-museums-attractions/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_Freethings.csv', skiprows=[1])

# Drop unnecessary columns
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image'], inplace=True)

# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_Freethings.csv', index=False)

print("stage1_data_Freethings.csv")

# Data Preprocessing

import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_Freethings.csv', skiprows=[1])
# print(df.columns)

punctuation = '!”$%&\’()*+,-./:;<=>?[\\]^_`{|}~•@©—'

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
df['text_cleaned_lemmatized'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned_lemmatized'] = df['text_cleaned_lemmatized'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)

df.to_csv('stage1_data_Freethings.csv', index=False)



# In[16]:


## 10 free things to do in Chicago this May

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
                data.append({'title': title, 'description': description,'personas': 'budget', 'Paragraph Index': idx, 'category': 'May free events'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/blog/free-cheap/10-free-things-to-do-in-chicago/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts, 'category': 'image'})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')

# Drop unnecessary columns (category_y)
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image', 'category_y'], inplace=True)



# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_MayEvents.csv', index=False)

print("Combined CSV file created successfully.")


# In[17]:


## 10 free things to do in Chicago this May cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_Freethings.csv', skiprows=[1])
# Web Scraping

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
                data.append({'title': title, 'description': description, 'personas': 'budget', 'Paragraph Index': idx, 'category': 'May free events'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/blog/free-cheap/10-free-things-to-do-in-chicago/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_MayEvents.csv', skiprows=[1])

# Drop unnecessary columns
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image'], inplace=True)

# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_MayEvents.csv', index=False)

print("stage1_data_MayEvents.csv")

# Data Preprocessing

import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_MayEvents.csv', skiprows=[1])
# print(df.columns)

punctuation = '!”$%&\’()*+,-./:;<=>?[\\]^_`{|}~•@©—'

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
df['text_cleaned_lemmatized'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned_lemmatized'] = df['text_cleaned_lemmatized'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)

df.to_csv('stage1_data_MayEvents.csv', index=False)



# In[18]:


## 5 free off-the-beaten-path things to do in Chicago

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
                data.append({'title': title, 'description': description,'personas': 'budget', 'Paragraph Index': idx, 'category': 'free hidden gems'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/free-and-cheap/5-free-off-the-beaten-path-things-to-do-in-chicago/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts, 'category': 'image'})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')

# Drop unnecessary columns (category_y)
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image', 'category_y'], inplace=True)



# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_HiddenGems.csv', index=False)

print("Combined CSV file created successfully.")


# In[19]:


## 5 free off-the-beaten-path things to do in Chicago cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_HiddenGems.csv', skiprows=[1])
# Web Scraping

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
                data.append({'title': title, 'description': description, 'personas': 'budget', 'Paragraph Index': idx, 'category': 'free hidden gems'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/free-and-cheap/5-free-off-the-beaten-path-things-to-do-in-chicago/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_HiddenGems.csv', skiprows=[1])

# Drop unnecessary columns
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image'], inplace=True)

# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_HiddenGems.csv', index=False)

print("stage1_data_HiddenGems.csv")

# Data Preprocessing

import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_HiddenGems.csv', skiprows=[1])
# print(df.columns)

punctuation = '!”$%&\’()*+,-./:;<=>?[\\]^_`{|}~•@©—'

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
df['text_cleaned_lemmatized'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned_lemmatized'] = df['text_cleaned_lemmatized'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)

df.to_csv('stage1_data_HiddenGems.csv', index=False)



# In[20]:


## Fun in the sun on Chicago’s beaches

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
                data.append({'title': title, 'description': description,'personas': 'budget', 'Paragraph Index': idx, 'category': 'beach'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/parks-outdoors/fun-in-the-sun-on-chicagos-beaches/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts, 'category': 'image'})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')

# Drop unnecessary columns (category_y)
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image', 'category_y'], inplace=True)



# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_beach.csv', index=False)

print("Combined CSV file created successfully.")


# In[22]:


## Fun in the sun on Chicago’s beaches cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_beach.csv', skiprows=[1])
# Web Scraping

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
                data.append({'title': title, 'description': description, 'personas': 'budget', 'Paragraph Index': idx, 'category': 'beach'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/parks-outdoors/fun-in-the-sun-on-chicagos-beaches/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_beach.csv', skiprows=[1])

# Drop unnecessary columns
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image'], inplace=True)

# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_beach.csv', index=False)

print("stage1_data_beach.csv")

# Data Preprocessing

import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_beach.csv', skiprows=[1])
# print(df.columns)

punctuation = '!”$%&\’()*+,-./:;<=>?[\\]^_`{|}~•@©—'

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
df['text_cleaned_lemmatized'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned_lemmatized'] = df['text_cleaned_lemmatized'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)

df.to_csv('stage1_data_beach.csv', index=False)



# In[24]:


## Your guide to the Chicago Riverwalk

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
                data.append({'title': title, 'description': description,'personas': 'budget', 'Paragraph Index': idx, 'category': 'riverwalk'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/free-and-cheap/chicago-riverwalk/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts, 'category': 'image'})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')

# Drop unnecessary columns (category_y)
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image', 'category_y'], inplace=True)



# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_riverwalk.csv', index=False)

print("Combined CSV file created successfully.")


# In[26]:


## Your guide to the Chicago Riverwalk cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_riverwalk.csv', skiprows=[1])
# Web Scraping

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
                data.append({'title': title, 'description': description, 'personas': 'budget', 'Paragraph Index': idx, 'category': 'riverwalk'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/free-and-cheap/chicago-riverwalk/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_riverwalk.csv', skiprows=[1])

# Drop unnecessary columns
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image'], inplace=True)

# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_riverwalk.csv', index=False)

print("stage1_data_riverwalk.csv")

# Data Preprocessing

import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_riverwalk.csv', skiprows=[1])
# print(df.columns)

punctuation = '!”$%&\’()*+,-./:;<=>?[\\]^_`{|}~•@©—'

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
df['text_cleaned_lemmatized'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned_lemmatized'] = df['text_cleaned_lemmatized'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)

df.to_csv('stage1_data_riverwalk.csv', index=False)



# In[27]:


## Chicago’s free summer music festivals

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
                data.append({'title': title, 'description': description,'personas': 'budget', 'Paragraph Index': idx, 'category': 'summer music festival'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/festivals-special-events/chicagos-free-summer-music-festivals/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts, 'category': 'image'})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')

# Drop unnecessary columns (category_y)
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image', 'category_y'], inplace=True)



# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_summermusicfestival.csv', index=False)

print("Combined CSV file created successfully.")


# In[29]:


## Chicago’s free summer music festivals cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_summermusicfestival.csv', skiprows=[1])
# Web Scraping

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
                data.append({'title': title, 'description': description, 'personas': 'budget', 'Paragraph Index': idx, 'category': 'summer music festival'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/festivals-special-events/chicagos-free-summer-music-festivals/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_summermusicfestival.csv', skiprows=[1])

# Drop unnecessary columns
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image'], inplace=True)

# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_summermusicfestival.csv', index=False)

print("stage1_data_summermusicfestival.csv")

# Data Preprocessing

import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_summermusicfestival.csv', skiprows=[1])
# print(df.columns)

punctuation = '!”$%&\’()*+,-./:;<=>?[\\]^_`{|}~•@©—'

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
df['text_cleaned_lemmatized'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned_lemmatized'] = df['text_cleaned_lemmatized'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)

df.to_csv('stage1_data_summermusicfestival.csv', index=False)



# In[30]:


## Public art in the Chicago Loop

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
                data.append({'title': title, 'description': description,'personas': 'budget', 'Paragraph Index': idx, 'category': 'public art'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/museums-art/public-art-in-the-chicago-loop/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts, 'category': 'image'})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')

# Drop unnecessary columns (category_y)
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image', 'category_y'], inplace=True)



# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_publicart.csv', index=False)

print("Combined CSV file created successfully.")


# In[31]:


## Public art in the Chicago Loop cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_publicart.csv', skiprows=[1])
# Web Scraping

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
                data.append({'title': title, 'description': description, 'personas': 'budget', 'Paragraph Index': idx, 'category': 'public art'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/museums-art/public-art-in-the-chicago-loop/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_publicart.csv', skiprows=[1])

# Drop unnecessary columns
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image'], inplace=True)

# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_publicart.csv', index=False)

print("stage1_data_publicart.csv")

# Data Preprocessing

import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_summermusicfestival.csv', skiprows=[1])
# print(df.columns)

punctuation = '!”$%&\’()*+,-./:;<=>?[\\]^_`{|}~•@©—'

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
df['text_cleaned_lemmatized'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned_lemmatized'] = df['text_cleaned_lemmatized'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)

df.to_csv('stage1_data_publicart.csv', index=False)



# In[37]:


## Free museum days in Chicago

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
                data.append({'title': title, 'description': description,'personas': 'budget', 'Paragraph Index': idx, 'category': 'free museum day'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/museums-art/free-museum-days-in-chicago/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts, 'category': 'image'})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')

# Drop unnecessary columns (category_y)
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image', 'category_y'], inplace=True)



# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_freemuseumday.csv', index=False)

print("Combined CSV file created successfully.")


# In[39]:


## Free museum days in Chicago cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_freemuseumday.csv', skiprows=[1])
# Web Scraping

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
                data.append({'title': title, 'description': description, 'personas': 'budget', 'Paragraph Index': idx, 'category': 'free museum day'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/museums-art/free-museum-days-in-chicago/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_freemuseumday.csv', skiprows=[1])

# Drop unnecessary columns
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image'], inplace=True)

# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_freemuseumday.csv', index=False)

print("stage1_data_freemuseumday.csv")

# Data Preprocessing

import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_freemuseumday.csv', skiprows=[1])
# print(df.columns)

punctuation = '!”$%&\’()*+,-./:;<=>?[\\]^_`{|}~•@©—'

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
df['text_cleaned_lemmatized'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned_lemmatized'] = df['text_cleaned_lemmatized'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)

df.to_csv('stage1_data_freemuseumday.csv', index=False)



# In[47]:


## 10 Chicago museums for $10 or less

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
                data.append({'title': title, 'description': description,'personas': 'budget', 'Paragraph Index': idx, 'category': 'cheap museum'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/blog/free-cheap/10-museums-under-10/'

# Scrape image URLs and paragraph counts
img_urls, p_counts = scrape_images_and_paragraphs(url)

# Scrape titles, descriptions, and paragraph indexes
title_description_data = scrape_titles_descriptions_and_indexes(url)

# Create DataFrames from scraped data
df_images = pd.DataFrame({'image_url': img_urls, 'Paragraphs Before Image': p_counts, 'category': 'image'})
df_titles = pd.DataFrame(title_description_data)

# Merge the DataFrames based on paragraph index
merged_df = pd.merge(df_titles, df_images, left_on='Paragraph Index', right_on='Paragraphs Before Image', how='left')

# Drop unnecessary columns (category_y)
merged_df.drop(columns=['Paragraph Index', 'Paragraphs Before Image', 'category_y'], inplace=True)



# Save the combined data to a CSV file
merged_df.to_csv('stage1_data_cheapmuseum.csv', index=False)

print("Combined CSV file created successfully.")


# In[51]:


## 10 Chicago museums for $10 or less cont.

import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_cheapmuseum1.csv', skiprows=[1])
# print(df.columns)

punctuation = '!”$%&\’()*+,-./:;<=>?[\\]^_`{|}~•@©—'

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
df['text_cleaned_lemmatized'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned_lemmatized'] = df['text_cleaned_lemmatized'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)

df.to_csv('stage1_data_cheapmuseum1.csv', index=False)



# In[56]:


## Combine all csv files


import os
import pandas as pd

# Specify the directory path where your CSV files are located
directory_path = '/Users/marianxu/Documents/ADSP34002/budget'

# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dataframes_list = []

# Loop through each CSV file and read its contents into a DataFrame
for file_name in csv_files:
    file_path = os.path.join(directory_path, file_name)
    df = pd.read_csv(file_path, encoding='latin-1')  # Try 'latin-1' encoding
    dataframes_list.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(dataframes_list, ignore_index=True)

# Write the combined DataFrame to a new CSV file
combined_file_path = '/Users/marianxu/Documents/ADSP34002/budget/Budget_combined_file.csv'
combined_df.to_csv(combined_file_path, index=False)

print(f"Combined {len(csv_files)} CSV files into '{combined_file_path}'")


# In[ ]:




