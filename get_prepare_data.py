from bs4 import BeautifulSoup
import requests
from os import path, makedirs


def download_save_mit_dataset():
    makedirs("data", exist_ok=True)

    url = 'https://groups.csail.mit.edu/sls/downloads/movie'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    for link in soup.find_all('a'):
        l = link.get('href')
        if '.bio' in l:
            file_path = path.join('data', l)
            if path.exists(file_path) and path.getsize(file_path) > 0:
                continue
            print(f"Downloading {l}")
            r = requests.get('/'.join((url, l)))
            with open(file_path, 'wb') as f:
                f.write(r.content)


if __name__ == '__main__':
    download_save_mit_dataset()
