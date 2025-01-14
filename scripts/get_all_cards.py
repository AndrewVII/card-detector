import requests
import os
import time


# Function to download an image from a URL and save it to a specified folder
def download_image(url, folder, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(folder, filename), "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
    else:
        print(f"Failed to download {filename}")


# Function to fetch card data from Scryfall API
def fetch_cards(page):
    url = f"https://api.scryfall.com/cards/search?q=game%3Apaper&order=set&page={page}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch page {page}")
        return None


folder = "data/cards"
if os.path.exists(folder):
    for file in os.listdir(folder):
        os.remove(os.path.join(folder, file))
    os.removedirs(folder)
os.makedirs(folder)

page = 1
while True:
    data = fetch_cards(page)
    if not data:
        break
    cards = data["data"]

    for card in cards:
        if "image_uris" in card:
            image_url = card["image_uris"]["normal"]
            raw_card_name = card["name"]
            # Remove special characters from the card name
            card_name = "".join(c for c in raw_card_name if c.isalnum() or c.isspace())
            filename = f"{card_name}.jpg"
            download_image(image_url, folder, filename)
            time.sleep(0.1)  # to avoid overwhelming the server with requests

    if not data["has_more"]:
        break

    page += 1

print("All cards downloaded successfully.")
