from bs4 import BeautifulSoup
import requests
from os import path, makedirs, listdir


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


def open_prepare():
    my_path = 'data'
    only_files = [path.join(my_path, f) for f in listdir(my_path) if
                  path.isfile(path.join(my_path, f)) and f.endswith('.bio')]

    files_dic = {}
    for b_f in only_files:
        with open(b_f, 'r')as f:
            files_dic.update({path.splitext(path.basename(b_f))[0]: f.readlines()})

    return files_dic


if __name__ == '__main__':
    # download_save_mit_dataset()
    files_dic = open_prepare()
    print('done')
