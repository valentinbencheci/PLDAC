import pandas as pd
import os
import json
import requests
import urllib.request
import logging
import datetime
from tqdm import tqdm

file_code = 0
log_path = '../log_' + datetime.datetime.now().date().strftime('%d_%m_%Y') + '.log'
logging.basicConfig(level=logging.INFO, filename=log_path, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

def get_page_per_country(country, page):
    """
    Download metadata content for bird recordings from Xeno-Canto for a specific country and page number

    @country: the country for which we download metadata content 
    @page: the current page to be downloaded
    @returns: the content downloaded
    """
    api_search = f"https://www.xeno-canto.org/api/2/recordings?query=cnt:{country}&page={page}"
    response = requests.get(api_search)
    if response.status_code == 200:
        response_payload = json.loads(response.content)
        return response_payload
    else:
        return None

def download_suite_from_country(country, original_path="../data/birds/original/"):
    """
    Download metadata content and bird recordings from Xeno-Canto for a specific country, saving the audio files locally
    
    @country: the country for which we download metadata content 
    @country_initial_payload: the initial downloaded payload for the country (1st page). We download all the other pages
    @returns: the content recordings (all pages, including the original one)
    """
    global file_code

    all_recordings = []
    pages = get_page_per_country(country, 1)["numPages"]

    if (not os.path.exists(original_path)):
        os.makedirs(original_path)
    
    for page in tqdm(range(1, pages+1)):
        payload = get_page_per_country(country, page)
        recordings = payload["recordings"]
        
        for file in recordings:
            urllib.request.urlretrieve(file["file"], original_path + "/" + str(file_code) + ".mp3")
            logging.info('File {} was downloaded'.format(file['file']))
            file_code += 1
        all_recordings.append(recordings)
    
    return all_recordings

def download_save_all_meta_for_country(countries=['France', 'Germany', 'Italy']):
    """
    Download and save metadata content for bird recordings from Xeno-Canto for a list of countries.

    @countries: list of countries for which we download content 
    """
    data = []
    for country in countries:
        data.extend(download_suite_from_country(country))
        logging.info('Bird recordings from Xeno-Canto were downloaded for the country {}'.format(country))

    data_df = pd.DataFrame.from_records(data)
    data_df.to_csv(f"../data/birds_original.csv", index=False)