#!/usr/bin/env python
# coding: utf-8

# In[30]:


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
nltk_stopwords = stopwords.words('english') # list of stopwords


# In[31]:


## Romantic Resturants/candlelight

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
                data.append({'title': title, 'description': description, 'Paragraph Index': idx, 'category': 'romantic restaurants'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/food-drink/romantic-candlelight-dining/'

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
merged_df.to_csv('stage1_data_CoupleRomantic.csv', index=False)

print("Combined CSV file created successfully.")


# In[32]:


## Romantic Resturants/candlelight cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_CoupleRomantic.csv', skiprows=[1])
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

df.to_csv('stage1_data_CoupleRomantic.csv', index=False)



# In[33]:


## Dinner Cruises

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
                data.append({'title': title, 'description': description, 'Paragraph Index': idx, 'category': 'dinner cruises'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/food-drink/chicago-dinner-cruises/'

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
merged_df.to_csv('stage1_data_CoupleCruises.csv', index=False)

print("Combined CSV file created successfully.")


# In[61]:


## Romantic Resturants/candlelight cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_CoupleCruises.csv', skiprows=[1])
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
df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)

df.to_csv('stage1_data_CoupleCruises.csv', index=False)



# In[34]:


## Waterfront Restaurants 

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
                data.append({'title': title, 'description': description, 'Paragraph Index': idx, 'category': 'waterfront restaurants'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/food-drink/head-to-the-beach-for-chicago-summer-eats/'

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
merged_df.to_csv('stage1_data_CoupleWaterfront.csv', index=False)

print("Combined CSV file created successfully.")


# In[39]:


## Waterfront Resturants cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_CoupleWaterfront.csv', skiprows=[1])
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
df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)

df.to_csv('stage1_data_CoupleWaterfront.csv', index=False)



# In[41]:


## Riverwalk Restaurants and Bars


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
                data.append({'title': title, 'description': description, 'Paragraph Index': idx, 'category': 'riverwalk restaurants'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/food-drink/chicago-riverwalk-restaurants-and-bars/'

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
merged_df.to_csv('stage1_data_CoupleRiverwalk.csv', index=False)

print("Combined CSV file created successfully.")


# In[42]:


## Riverwalk Restaurants and Bars cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_CoupleRiverwalk.csv', skiprows=[1])
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
df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)

df.to_csv('stage1_data_CoupleRiverwalk.csv', index=False)



# In[43]:


## Outdoor Dining

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
                data.append({'title': title, 'description': description, 'Paragraph Index': idx, 'category': 'outdoor dining'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/food-drink/chicago-patios-and-rooftop-bars-with-spectacular-views/'

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
merged_df.to_csv('stage1_data_CoupleOutdoor.csv', index=False)

print("Combined CSV file created successfully.")


# In[44]:


## Outdoor Dining cont.

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_CoupleOutdoor.csv', skiprows=[1])
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
df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)


df.to_csv('stage1_data_CoupleOutdoor.csv', index=False)


# In[45]:


## Picnic Spots

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
                data.append({'title': title, 'description': description, 'Paragraph Index': idx, 'category': 'picnic'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/blog/arts-culture-entertainment/seven-chicago-picnic-spots/'

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
merged_df.to_csv('stage1_data_CouplePicnic.csv', index=False)

print("Combined CSV file created successfully.")


# In[46]:


## Outdoor Dining cont.

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_CouplePicnic.csv', skiprows=[1])
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
df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)


df.to_csv('stage1_data_CouplePicnic.csv', index=False)


# In[47]:


## Indoor Gardens

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
                data.append({'title': title, 'description': description, 'Paragraph Index': idx, 'category': 'indoor gardens'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/parks-outdoors/chicago-conservatories-and-winter-gardens/'

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
merged_df.to_csv('stage1_data_CoupleIndoorGardens.csv', index=False)

print("Combined CSV file created successfully.")


# In[48]:


## Indoor Gardens cont. 

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_CoupleIndoorGardens.csv', skiprows=[1])
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
df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)


df.to_csv('stage1_data_CoupleIndoorGardens.csv', index=False)


# In[49]:


## Waterfronts 

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
                data.append({'title': title, 'description': description, 'Paragraph Index': idx, 'category': 'waterfronts'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/parks-outdoors/2-days-2-chicago-waterfronts/'

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
merged_df.to_csv('stage1_data_CoupleWaterfronts.csv', index=False)

print("Combined CSV file created successfully.")


# In[50]:


## Waterfronts cont. 

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_CoupleWaterfronts.csv', skiprows=[1])
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
df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)


df.to_csv('stage1_data_CoupleWaterfronts.csv', index=False)


# In[51]:


## Beaches

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
                data.append({'title': title, 'description': description, 'Paragraph Index': idx, 'category': 'beaches'})

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
merged_df.to_csv('stage1_data_CoupleBeaches.csv', index=False)

print("Combined CSV file created successfully.")


# In[52]:


## Beaches cont.

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_CoupleBeaches.csv', skiprows=[1])
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
df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)


df.to_csv('stage1_data_CoupleBeaches.csv', index=False)


# In[53]:


## Millennium Park

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
                data.append({'title': title, 'description': description, 'Paragraph Index': idx, 'category': 'MillenniumPark'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/parks-outdoors/millennium-park-campus/'

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
merged_df.to_csv('stage1_data_CoupleMillenniumPark.csv', index=False)

print("Combined CSV file created successfully.")


# In[54]:


## Millennium Park cont.

# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_CoupleMillenniumPark.csv', skiprows=[1])
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
df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)


df.to_csv('stage1_data_CoupleMillenniumPark.csv', index=False)


# In[55]:


## Date Night Ideas


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
                data.append({'title': title, 'description': description, 'Paragraph Index': idx, 'category': 'date night ideas'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/itineraries/3-great-chicago-date-ideas/'

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
merged_df.to_csv('stage1_data_CoupleDateNightIdeas.csv', index=False)

print("Combined CSV file created successfully.")


# In[56]:


## Date Night Ideas cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_CoupleDateNightIdeas.csv', skiprows=[1])
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
df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)


df.to_csv('stage1_data_CoupleDateNightIdeas.csv', index=False)


# In[57]:


## Romantic Weekend


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
                data.append({'title': title, 'description': description, 'Paragraph Index': idx, 'category': 'romantic weekend'})

        return data
    else:
        # If the request was not successful, print an error message
        print("Failed to fetch the URL:", url)
        return None

# URL of the webpage to scrape
url = 'https://www.choosechicago.com/articles/itineraries/48-hour-romance/'

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
merged_df.to_csv('stage1_data_CoupleRomanticWeekend.csv', index=False)

print("Combined CSV file created successfully.")


# In[58]:


## Romantic Weekend cont.


# Read the CSV file into a DataFrame, skipping the first row
df = pd.read_csv('stage1_data_CoupleRomanticWeekend.csv', skiprows=[1])
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
df['text_cleaned'] = df['text_cleaned'].apply(remove_stopwords)
df['text_cleaned'] = df['text_cleaned'].apply(lemmatize_word)
# df['text_cleaned_token'] = df['text_cleaned_token'].apply(remove_stopwords)


df.to_csv('stage1_data_CoupleRomanticWeekend.csv', index=False)


# In[62]:


## Combine all csv files


import os
import pandas as pd

# Specify the directory path where your CSV files are located
directory_path = '/Users/marianxu/Documents/ADSP34002'

# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dataframes_list = []

# Loop through each CSV file and read its contents into a DataFrame
for file_name in csv_files:
    file_path = os.path.join(directory_path, file_name)
    df = pd.read_csv(file_path)
    dataframes_list.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(dataframes_list, ignore_index=True)

# Write the combined DataFrame to a new CSV file
combined_file_path = '/Users/marianxu/Documents/ADSP34002/Couple_combined_file.csv'
combined_df.to_csv(combined_file_path, index=False)

print(f"Combined {len(csv_files)} CSV files into '{combined_file_path}'")



# In[ ]:




