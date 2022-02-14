from bs4 import BeautifulSoup
import urllib
from urllib.request import urlopen
import requests
from os.path import basename
import pandas as pd
import sys


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
