from bs4 import BeautifulSoup
import urllib
from urllib.request import urlopen
import requests
from os.path import basename
import pandas as pd


def download_data():
    url = 'https://datasets.imdbws.com/'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    for link in soup.find_all('a'):
        url = link.get('href')
        if url.endswith('.gz'):
            urllib.request.urlretrieve(url, basename(url))

    print('done')


def main():
    data = pd.read_csv(r'E:\pv\media_manager\data\name.basics.tsv\data.tsv', sep='\t')
    print('done')


if __name__ == '__main__':
    main()
