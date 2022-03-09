import sys
import logging
from os import rename
import pandas as pd
from simpletransformers.ner import NERModel, NERArgs
import download_filles as df_module
import get_prepare_mit_files as gp_mit


def train_save_model(model_type, model_name, train_data, eval_data, labels, model_args):
    model_bert = NERModel(model_type=model_type, model_name=model_name, labels=labels, args=model_args,
                          use_cuda=use_cuda)
    model_bert.train_model(train_data=train_data, eval_data=eval_data)


def input_frac_data():
    while True:
        try:
            frac = float(input('What fraction of data do you want to process (0-1)?'))
            if 0 < frac <= 1:
                break
            else:
                print('enter a number between 0-1')
        except Exception as ex:
            logger.error(ex)
            raise Exception(ex)
    return frac


def train():
    # region load & prepare data
    logger.info('Load and prepare mit movie corpus dataset...')
    df_train, df_test, label = gp_mit.prepare_data(logger)
    logger.info(f'loading {df_train.shape[0]} train and {df_test.shape[0]} test data.')

    frac = input_frac_data()
    logger.info(f'train models with {frac * 100} percent of data')
    df_train = df_train.sample(frac=frac, ignore_index=True)
    df_test = df_test.sample(frac=frac, ignore_index=True)
    # endregion

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

    # region train models
    models = {'bert': 'bert-base-cased',
              'distilbert': 'distilbert-base-cased',
              'roberta': 'roberta-base'}

    for model_type, model_name in models.items():
        logger.info(f'train and evaluate {model_type}')
        try:
            train_save_model(model_type, model_name, df_train, df_test, label, model_args)
            rename('outputs', f'{model_type}_outputs')
        except Exception as ex:
            logger.error(ex)
            raise Exception(ex)
    # endregion

    logger.info(f'training models complete with {df_train.shape[0]} train tokens, '
                f'{len(df_train["sentence_id"].unique())} train sentences, '
                f'{df_test.shape[0]} tests tokens and '
                f'{len(df_test["sentence_id"].unique())} test sentences')


def test(sentences_list: list) -> pd.DataFrame:
    models = ['bert', 'distilbert', 'roberta']

    sentence_li = []
    for m in models:
        logger.info(f'loading {m} and prediction...')
        try:
            model_bert = NERModel(m, f'{m}_outputs/best_model/', use_cuda=use_cuda)
            prediction, model_output = model_bert.predict(sentences_list)
        except Exception as ex:
            logger.error(f'could not load {m}. Error: {ex}')
            raise Exception(ex)

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


def main():
    if not gp_mit.check_mit_exists():
        logger.warning("the mit movie corpus files do not exist. you need to download them in the first run!!!")
        if df_module.query_yes_no(question='download mit movie corpus dataset?'):
            logger.info('downloading mit movie corpus')
            gp_mit.download_save_mit_dataset()
        else:
            logger.warning('mit movie corpus dataset needs in first run! closing program...')
            sys.exit(0)

    if df_module.query_yes_no(question='train bert models?'):
        train()

    if df_module.query_yes_no(question='test bert models?'):
        sentences = ['I want a movie from Christopher Nolan in 2015',
                     'Show me the best America serials of 90s',
                     'The golden globe winning dram movies in history',
                     'Episode 2 season 7 of friends serial',
                     'Best new action movies of Vin Diesel']
        logger.info(f'testing models for {len(sentences)} sentences...')
        df_results = test(sentences)

        logger.info('test finished. write the results in results.csv')
        df_results.to_csv('results.csv')


def check_cuda():
    import torch
    if torch.cuda.is_available():
        return True
    else:
        logger.warning('cuda is not available on this system!')
        return False


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    use_cuda = check_cuda()

    main()
