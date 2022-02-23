from bs4 import BeautifulSoup
import urllib
from urllib.request import urlopen
import requests
from clint.textui import progress
from os.path import basename
from os import path
import pandas as pd
import sys
from os import makedirs
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def download_save(url: str, file_dir: str, extensions: list):
    # imdb_path = "data/imdb"
    makedirs(file_dir, exist_ok=True)

    # url = 'https://datasets.imdbws.com/'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    for link in soup.find_all('a'):
        l = link.get('href')
        if l.endswith(tuple(extensions)):
            file_path = path.join(file_dir, basename(l))
            if path.exists(file_path) and path.getsize(file_path) > 0:
                continue
            print(f"Downloading {l}")
            download_url(url=l, output_path=file_path)


def download_imdb():
    makedirs("data", exist_ok=True)

    url = 'https://datasets.imdbws.com/'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    for link in soup.find_all('a'):
        l = link.get('href')
        if l.endswith('.gz'):
            urllib.request.urlretrieve(url, basename(url))

    print('done')


def main():
    data = pd.read_csv(r'E:\pv\media_manager\data\name.basics.tsv\data.tsv', sep='\t')
    print('done')


if __name__ == '__main__':
    # main()
    download_save_imdb_dataset()
