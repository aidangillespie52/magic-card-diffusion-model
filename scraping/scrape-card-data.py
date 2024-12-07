import aiohttp
import aiofiles
import asyncio
import os
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

INVALID_CHARS = [',', '"', '?', '=', '+', '~', '%', '<', '>', ':', '|', '/', '*', '&']
WEBSITE_LINK = 'https://scryfall.com/sets/?lang=en'

async def clean_filename(name):
    # Replace invalid characters with an empty string
    for char in INVALID_CHARS:
        name = name.replace(char, '')
    return name

async def fetch(session, url):
    # Perform an HTTP GET request asynchronously
    async with session.get(url) as response:
        return await response.text()

async def download_image(session, img_link, img_file_path):
    # Download image and save asynchronously
    async with session.get(img_link) as response:
        content = await response.read()
        async with aiofiles.open(img_file_path, 'wb') as f:
            await f.write(content)

async def main():
    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        # Get the response from the website asynchronously
        response = await fetch(session, WEBSITE_LINK)
        soup = BeautifulSoup(response, 'html.parser')
        table = soup.find('table')
        trs = table.findAll('tr')

        magic_sets = set()

        # Iterate through the table rows and get the name and link of each set
        for tr in trs[1:]:
            name = tr.td.a.text.strip()
            link = tr.td.a['href']
            if 'https://scryfall.com/sets/' in link and '/en' in link:
                magic_sets.add((name, link))

        # For each magic set, download images asynchronously
        tasks = []
        for name, link in tqdm(magic_sets):
            name = await clean_filename(name)

            # Create directory for magic_set
            set_directory = '../data/' + name
            os.makedirs(set_directory, exist_ok=True)

            # Get all images from the set asynchronously
            response = await fetch(session, link)
            soup = BeautifulSoup(response, 'html.parser')
            main_card_holder = soup.find('div', attrs={'class': 'card-grid'})
            images = main_card_holder.find_all('img')

            # Add tasks to download images
            for img in images:
                img_link = img['src']
                img_name = img['title']
                img_name = await clean_filename(img_name)

                # Construct the file path
                img_file_path = os.path.join(set_directory, img_name + '.jpg')
                tasks.append(download_image(session, img_link, img_file_path))

        # Run all download tasks concurrently
        await asyncio.gather(*tasks)

# Run the asyncio event loop
if __name__ == '__main__':
    asyncio.run(main())
