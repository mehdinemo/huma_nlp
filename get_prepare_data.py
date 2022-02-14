import pandas as pd
import io
from bs4 import BeautifulSoup
import requests
from os import path, makedirs, listdir, rename
from simpletransformers.ner import NERModel, NERArgs


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


def open_files():
    my_path = 'data'
    only_files = [path.join(my_path, f) for f in listdir(my_path) if
                  path.isfile(path.join(my_path, f)) and f.endswith('.bio')]

    files_dic = {}
    for b_f in only_files:
        with open(b_f, 'r') as f:
            files_dic.update({path.splitext(path.basename(b_f))[0]: f.readlines()})

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


def train_eval_save_model(df_train, df_test, label):
    # region model args
    model_args = NERArgs()
    model_args.num_train_epochs = 2
    model_args.learning_rate = 1e-4
    model_args.overwrite_output_dir = True
    model_args.train_batch_size = 32
    model_args.eval_batch_size = 8
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 1000
    model_args.evaluate_during_training_verbose = True
    # endregion

    model_bert = NERModel('bert', 'bert-base-cased', labels=label, args=model_args, use_cuda=False)
    model_bert.train_model(df_train, eval_data=df_test, output_dir='bert_output/')
    rename('outputs', 'bert_best')

    model_distilbert = NERModel('distilbert', 'distilbert-base-cased', labels=label, args=model_args, use_cuda=False)
    model_distilbert.train_model(df_train, eval_data=df_test, output_dir='distilbert_output/')
    rename('outputs', 'distilbert_best')

    model_roberta = NERModel('roberta', 'roberta-base', labels=label, args=model_args, use_cuda=False)
    model_roberta.train_model(df_train, eval_data=df_test, output_dir='roberta_output')
    rename('outputs', 'roberta_best')

    result_bert, model_outputs, preds_list = model_bert.eval_model(df_test)
    result_distilbert, model_outputs, preds_list = model_distilbert.eval_model(df_test)
    result_roberta, model_outputs, preds_list = model_roberta.eval_model(df_test)

    df = pd.DataFrame([result_bert, result_roberta, result_distilbert],
                      index=['result_bert', 'result_roberta', 'result_distilbert'])

    df['test_token_count'] = df_test.shape[0]
    df['test_sentence_count'] = len(df_test['sentence_id'].unique())

    df['train_token_count'] = df_train.shape[0]
    df['train_sentence_count'] = len(df_train['sentence_id'].unique())

    df.to_csv('eval_results.csv')


def prepare_data():
    files_dic = open_files()

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


def main():
    if train:
        df_train, df_test, label = prepare_data()

        df_train = df_train.sample(frac=0.05, ignore_index=True)
        df_test = df_test.sample(frac=0.05, ignore_index=True)
        train_eval_save_model(df_train, df_test, label)

    model_bert = NERModel('bert', 'outputs/best_model/', use_cuda=False)
    prediction, model_output = model_bert.predict(
        ["What 2011 animated movie starred the voices of johnny deep and rahul poddar"])

    print('done')


if __name__ == '__main__':
    train = False
    download_save_mit_dataset()
    main()
