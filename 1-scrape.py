# 1. download PDFs for Field museum news from biodiversity heritage library

import requests
import xml.etree.ElementTree as ET
import urllib

import streamlit as st

BHL_KEY = st.secrets['BHL_KEY']

def get_urls(title: str, year: int):
    """
    title = Title name to search for (e.g., 'Field Museum news')
    year = Year of search
    """
    
    # Convert year to string
    year = str(year)

    # Make title URL-friendly
    title = urllib.parse.quote(title.lower())
    url = 'https://www.biodiversitylibrary.org/api3?op=PublicationSearchAdvanced&title=' + title + '&year=' + year + '&format=xml&apikey=' + BHL_KEY

    # Get XML for search item
    page = requests.get(url)
    root = ET.fromstring(page.content)

    if (len(root.find('Result')) > 0):
        
        # Extract item ID    
        item_id = root.find('Result/Publication/ItemID').text
        
        # Create a new call to get item metadata (i.e. PDF url is what we're after)
        url = 'https://www.biodiversitylibrary.org/api3?op=GetItemMetadata&id=' + item_id + '&apikey=' + BHL_KEY
    
        page = requests.get(url)
        root = ET.fromstring(page.content)

        # Extract PDF url
        pdf_url = root.find('Result/Item/ItemPDFUrl').text
        
        # Download PDF to folder
        basename = pdf_url.split('/')[-1]
        print("Downloading PDF " + basename + "...")
        
        output_path = "pdfs/"+basename+".pdf"

        if not os.path.exists(output_path):
            urllib.request.urlretrieve(pdf_url, output_path)
            print("Downloaded PDF successfully")
        else:
            print(f"File {basename}.pdf already exists, skipping download")

    else:
        print("Publication doesn't exist.")
        pass

for year in range(1930, 1992):
    get_urls(title='Field Museum news', year=year)


# 2. 

