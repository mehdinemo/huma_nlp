import io
import pandas as pd
from os import path, makedirs, listdir
import download_filles as df_module

_mit_path = 'data/mit_movie_corpus'


def check_mit_exists():
    mit_files = ['engtrain.bio', 'engtest.bio']
    for f in mit_files:
        filename = path.join(_mit_path, f)
        if not path.isfile(filename):
            return False

    return True


def download_save_mit_dataset():
    makedirs(_mit_path, exist_ok=True)

    url = "https://groups.csail.mit.edu/sls/downloads/movie"
    df_module.download_save(url=url, file_dir=_mit_path, extensions=['bio'])


def open_mit_files(logger):
    try:
        only_files = [path.join(_mit_path, f) for f in listdir(_mit_path) if
                      path.isfile(path.join(_mit_path, f)) and f.endswith('.bio')]
    except FileNotFoundError as ex:
        logger.error(f'FileNotFoundError: {ex}')
        raise FileNotFoundError(ex)

    files_dic = {}
    for b_f in only_files:
        try:
            with open(b_f, 'r') as f:
                files_dic.update({path.splitext(path.basename(b_f))[0]: f.readlines()})
        except Exception as ex:
            logger.error(f'Loading file error: {ex}')
            raise Exception(ex)

    return files_dic


def mit_list2df(df_list):
    sentense_id = 0
    new_df_list = []
    for line in df_list:
        if line == '\n':
            sentense_id += 1
        else:
            new_df_list.append(str(sentense_id) + '\t' + line)

    return new_df_list


def prepare_data(logger):
    files_dic = open_mit_files(logger)

    tmp_list = mit_list2df(files_dic['engtrain'])
    df_train = pd.read_csv(io.StringIO(''.join(tmp_list)), delim_whitespace=True, header=None,
                           names=['sentence_id', 'labels', 'words'])
    df_train = df_train[df_train['words'].notnull()]

    tmp_list = mit_list2df(files_dic['engtest'])
    df_test = pd.read_csv(io.StringIO(''.join(tmp_list)), delim_whitespace=True, header=None,
                          names=['sentence_id', 'labels', 'words'])
    df_test = df_test[df_test['words'].notnull()]

    label = df_train["labels"].unique().tolist()

    return df_train, df_test, label
