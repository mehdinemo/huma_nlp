import pandas as pd
import logging
import io
from os import path, makedirs, listdir, rename
from simpletransformers.ner import NERModel, NERArgs
import download_filles as df
from config import config


def download_save_mit_dataset():
    mit_path = config['mit_path']
    makedirs(mit_path, exist_ok=True)

    url = config['mit_url']
    df.download_save(url=url, file_dir=mit_path, extensions=['bio'])


def open_mit_files():
    mit_path = config['mit_path']
    try:
        only_files = [path.join(mit_path, f) for f in listdir(mit_path) if
                      path.isfile(path.join(mit_path, f)) and f.endswith('.bio')]
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


def prepare_data():
    files_dic = open_mit_files()

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


def train_save_model(model_type, model_name, train_data, eval_data, labels, model_args, use_cuda):
    model_bert = NERModel(model_type=model_type, model_name=model_name, labels=labels, args=model_args,
                          use_cuda=use_cuda)
    model_bert.train_model(train_data=train_data, eval_data=eval_data)


def train():
    logger.info('Load and prepare data')
    df_train, df_test, label = prepare_data()

    # region model args
    model_args = NERArgs()
    model_args.num_train_epochs = 2
    model_args.learning_rate = 1e-4
    model_args.overwrite_output_dir = True
    model_args.train_batch_size = 8
    model_args.eval_batch_size = 8
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 1000
    model_args.evaluate_during_training_verbose = True
    # endregion

    models = {'bert': 'bert-base-cased',
              'distilbert': 'distilbert-base-cased',
              'roberta': 'roberta-base'}

    for model_type, model_name in models.items():
        logger.info(f'train and evaluate {model_type}')
        try:
            train_save_model(model_type, model_name, df_train, df_test, label, model_args, use_cuda=True)
            rename('outputs', f'{model_type}_outputs')
        except Exception as ex:
            logger.error(ex)
            raise Exception(ex)

    data_cnt = {}
    data_cnt.update({'test_token_count': df_test.shape[0]})
    data_cnt.update({'test_sentence_count': len(df_test['sentence_id'].unique())})
    data_cnt.update({'train_token_count': df_train.shape[0]})
    data_cnt.update({'train_sentence_count': len(df_train['sentence_id'].unique())})

    with open('data/data_cnt.txt', 'w') as f:
        for key, value in data_cnt.items():
            f.write(f'{key} = {value}\n')


def test():
    models = ['bert', 'distilbert', 'roberta']

    sentences = ['What 2011 animated movie starred the voices of johnny deep and rahul poddar',
                 'I want a movie from Christopher Nolan in 2015',
                 'Show me the best America serials of 90s',
                 'The golden globe winning dram movies in history',
                 'Episode 2 season 7 of friends serial',
                 'Best new action movies of Vin Diesel']
    sentence_li = []
    for m in models:
        model_bert = NERModel(m, f'All Models/{m}_outputs/best_model/', use_cuda=True)
        prediction, model_output = model_bert.predict(sentences)

        for i in range(len(prediction)):
            kl = [list(p.keys())[0] for p in prediction[i]]
            vl = [list(p.values())[0] for p in prediction[i]]
            df_p = pd.DataFrame({'world': kl, 'label': vl})
            df_p['sentence'] = i
            df_p['model'] = m
            sentence_li.append(df_p)

    df = pd.concat(sentence_li, ignore_index=True)
    new_df = df.pivot(index=['world', 'sentence'], columns='model')
    new_df.fillna(0, inplace=True)
    new_df.reset_index(inplace=True)
    new_df.sort_values(by=['sentence'], inplace=True)
    new_df.reset_index(drop=True, inplace=True)

    return new_df


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()
    # if df.query_yes_no(question='downloading mit movie corpus dataset?'):
    #     download_save_mit_dataset()
    #

    train()
    # df = test()
