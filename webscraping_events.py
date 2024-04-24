import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_events_text(url, max_pages=1):
    event_data_text = []

    for page_num in range(1, max_pages + 1):
        # Construct URL for current page
        page_url = url
        if page_num > 1:
            # Assuming pagination uses URL parameters (adjust based on website structure)
            page_url = f"{url}?page={page_num}"

        # Send request and parse HTML
        response = requests.get(page_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find event elements
            events = soup.find_all('div', class_='card-body')

            # Extract event details for each element
            for event in events:
                category_element = event.find('h6')
                category = category_element.text.strip() if category_element else None

                title_element = event.find('h4', class_='card-title')
                title = title_element.text.strip() if title_element else None

                text_element = event.find('p')
                text = text_element.text.strip() if text_element else None

                # Only append non-blank event data
                if category or title or text:
                    event_data_text.append({
                        'category': category,
                        'title': title,
                        'description': text
                    })
        else:
            print(f"Error getting data for page {page_num}: {response.status_code}")

    return event_data_text

def scrape_events_metadata(url, max_pages=1):
    event_data_metadata = []

    for page_num in range(1, max_pages + 1):
        # Construct URL for current page
        page_url = url
        if page_num > 1:
            # Assuming pagination uses URL parameters (adjust based on website structure)
            page_url = f"{url}?page={page_num}"

        # Send request and parse HTML
        response = requests.get(page_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find event elements
            events = soup.find_all('div', class_='card-body h-100 px-md-4 py-xl-5')

            # Extract event details for each element
            for event in events:
                startd_element = event.find('span', class_='tribe-event-date-start')
                startd = startd_element.text.strip() if startd_element else None

                endd_element = event.find('span', class_='tribe-event-date-end')
                endd = endd_element.text.strip() if endd_element else None

                time_elements = event.find_all('span', class_='tribe-event-time')

                # Extract start time from the first element
                start_time = time_elements[0].text.strip() if time_elements else None

                # Extract end time from the second element if available
                if len(time_elements) > 1:
                    end_time = time_elements[1].text.strip()
                else:
                    end_time = None

                venue_element = event.find('b')
                venue = venue_element.text.strip() if venue_element else None

                # Find the <br> tag after the venue
                address_element = venue_element.find_next_sibling('br')

                # Get the text after <br> if address_element exists and has a next sibling
                if address_element and address_element.next_sibling:
                    address = address_element.next_sibling.strip()
                else:
                    address = None

                event_data_metadata.append({
                    'start_date': startd,
                    'start_time': start_time,
                    'end_date': endd,
                    'end_time': end_time,
                    'location': venue,
                    'address': address
                })
        else:
            print(f"Error getting data for page {page_num}: {response.status_code}")

    return event_data_metadata

def scrape_events_image(url, max_pages=1):
    event_data_image = []

    for page_num in range(1, max_pages + 1):
        # Construct URL for current page
        page_url = url
        if page_num > 1:
            # Assuming pagination uses URL parameters (adjust based on website structure)
            page_url = f"{url}?page={page_num}"

        # Send request and parse HTML
        response = requests.get(page_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find event elements
            events = soup.find_all('div', class_='event-image col-12 p-0 col-md-3')

            # Extract event details for each element
            for event in events:
                img_element = event.find('img')
                img = img_element.get('data-src') if img_element else None

                event_data_image.append({
                  'image_url': img
                })
        else:
            print(f"Error getting data for page {page_num}: {response.status_code}")

    return event_data_image

# Example usage
url = 'https://www.choosechicago.com/events/'
max_pages = 10  # Adjust as needed

event_data_text = scrape_events_text(url, max_pages)
event_data_metadata = scrape_events_metadata(url, max_pages)
event_data_image = scrape_events_image(url, max_pages)

# Combine the three lists of event data
combined_event_data = []
for text_data, metadata, image_data in zip(event_data_text, event_data_metadata, event_data_image):
    combined_event_data.append({**text_data, **metadata, **image_data})

# Convert the combined event data into a DataFrame
events_df = pd.DataFrame(combined_event_data)

if events_df.empty:
    print("No events found.")
else:
    # Save to CSV (optional)
    events_df.to_csv('scraped_events.csv', index=False)
    print("Events saved to scraped_events.csv")
    print(events_df.head())  # Print the first few rows

