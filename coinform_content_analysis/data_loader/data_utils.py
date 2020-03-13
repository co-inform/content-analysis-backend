import json
import os
import pickle
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger
from torchnlp.datasets.dataset import Dataset
from coinform_content_analysis.preprocessing import handle_twitter_specific_tags

from coinform_content_analysis.feat_extractor import TextFeatures

'''Retrieving code is inspired by https://github.com/UKPLab/mdl-stance-robustness'''
SEMEVAL_PATH = Path('data/SemEval2019Task7')
SEMEVAL_TASKA = 'subtaskaenglish'
SEMEVAL_TASKB = 'subtaskbenglish'
SEMEVAL_FEATS_TRAIN = Path('data/SemEval2019Task7/features/train')
SEMEVAL_FEATS_TEST = Path('data/SemEval2019Task7/features/test')
SEMEVAL_FEATS_DEV = Path('data/SemEval2019Task7/features/dev')

TEXT_FEAT_EXTRACTOR = TextFeatures()

FEATURES_MAP = {
    'bias': TEXT_FEAT_EXTRACTOR.get_bias_feats,
    'moral_foundation': TEXT_FEAT_EXTRACTOR.get_moral_foundation_feats,
    'complexity': TEXT_FEAT_EXTRACTOR.get_complexity_feats,
    'linked_feat': TEXT_FEAT_EXTRACTOR.get_linked_feat,
    'structural': TEXT_FEAT_EXTRACTOR.get_structural_feats,
    'sentiment': TEXT_FEAT_EXTRACTOR.get_sentiment_feats
}


def count_classes(y):
    """
    Returns a dictionary with class balances for each class (absolute and relative)
    :param y: gold labels as array of int
    """
    count_dict = Counter([l for l in y])
    return {l: str(c) + " ({0:.2f}".format((c / len(y)) * 100) + "%)" for l, c in count_dict.items()}


def read_and_tokenize_json(file_name, task_type, tweet_dict):
    X, y = [], []

    with open(SEMEVAL_PATH / file_name, "r") as in_f:
        split_dict = json.load(in_f)[task_type]

        for tweet_id, label in split_dict.items():
            try:
                X.append(tweet_dict[tweet_id])
            except:
                continue
            y.append(label)

    return X, y


def read_and_tokenize_zip(folder_path, set_file, task_type, tweet_dict):
    X, y = [], []

    with zipfile.ZipFile(SEMEVAL_PATH / folder_path, 'r') as z:
        with z.open("rumoureval-2019-training-data/" + set_file) as in_f:
            split_dict = json.load(in_f)[task_type]

            for tweet_id, label in split_dict.items():
                try:
                    X.append(tweet_dict[tweet_id])
                except:
                    continue
                y.append(label)

        return X, y


def parse_tweets(folder_path):
    # create a dict with key = reply_tweet_id and values = source_tweet_id, source_tweet_text, reply_tweet_text
    tweet_dict = {}
    with zipfile.ZipFile(folder_path, 'r') as z:
        for filename in z.namelist():
            if not filename.lower().endswith(".json") or filename.rsplit("/", 1)[1] in ['raw.json', 'structure.json',
                                                                                        'dev-key.json',
                                                                                        'train-key.json']:
                continue
            with z.open(filename) as f:
                data = f.read()
                d = json.loads(data.decode("ISO-8859-1"))

                if "data" in d.keys():  # reddit
                    if "body" in d['data'].keys():  # reply
                        tweet_dict[d['data']['id']] = d['data']['body']
                    elif "children" in d['data'].keys() and isinstance(d['data']['children'][0], dict):
                        tweet_dict[d['data']['children'][0]['data']['id']] = d['data']['children'][0]['data']['title']
                    else:  # source
                        try:
                            tweet_dict[d['data']['children'][0]] = ""
                        except Exception as e:
                            print(e)

                if "text" in d.keys():  # twitter
                    tweet_dict[str(d['id'])] = d['text']

    return tweet_dict


def extract_rumoureval2019_features():
    if not SEMEVAL_FEATS_TRAIN.exists():
        os.makedirs(SEMEVAL_FEATS_TRAIN)
    if not SEMEVAL_FEATS_DEV.exists():
        os.makedirs(SEMEVAL_FEATS_DEV)
    if not SEMEVAL_FEATS_TEST.exists():
        os.makedirs(SEMEVAL_FEATS_TEST)

    logger.debug('Retrieving train/dev data on Task A')
    tweet_dict = parse_tweets(SEMEVAL_PATH / 'rumoureval-2019-training-data.zip')

    X_stance_train, y_stance_train = read_and_tokenize_zip('rumoureval-2019-training-data.zip', 'train-key.json',
                                                           SEMEVAL_TASKA, tweet_dict)
    X_stance_dev, y_stance_dev = read_and_tokenize_zip('rumoureval-2019-training-data.zip', 'dev-key.json',
                                                       SEMEVAL_TASKA, tweet_dict)

    logger.debug('Retrieving train/dev data on Task B')
    X_veracity_train, y_veracity_train = read_and_tokenize_zip('rumoureval-2019-training-data.zip', 'train-key.json',
                                                               SEMEVAL_TASKB, tweet_dict)
    X_veracity_dev, y_veracity_dev = read_and_tokenize_zip('rumoureval-2019-training-data.zip', 'dev-key.json',
                                                           SEMEVAL_TASKB, tweet_dict)

    print(y_veracity_dev)

    logger.debug('Retrieving test data on Task A')
    tweet_dict = parse_tweets(SEMEVAL_PATH / "rumoureval-2019-test-data.zip")
    X_stance_test, y_stance_test = read_and_tokenize_json("final-eval-key.json", SEMEVAL_TASKA, tweet_dict)

    logger.debug('Retrieving test data on Task B')
    X_veracity_test, y_veracity_test = read_and_tokenize_json("final-eval-key.json", SEMEVAL_TASKB, tweet_dict)

    # class balance per set
    logger.info('=============Dataset Distribution of Subtask A===================')
    logger.info("\nClass balance for train: " + str(count_classes(y_stance_train)))
    logger.info("Class balance for dev: " + str(count_classes(y_stance_dev)))
    logger.info("Class balance for test: " + str(count_classes(y_stance_test)))
    logger.info("Class balance for all data: " + str(count_classes(y_stance_train + y_stance_dev + y_stance_test)))

    # class balance per set
    logger.info('=============Dataset Distribution of Subtask B===================')
    logger.info("\nClass balance for train: " + str(count_classes(y_veracity_train)))
    logger.info("Class balance for dev: " + str(count_classes(y_veracity_dev)))
    logger.info("Class balance for test: " + str(count_classes(y_veracity_test)))
    logger.info(
        "Class balance for all data: " + str(count_classes(y_veracity_train + y_veracity_dev + y_veracity_test)))

    # ======================== Save the rows for transformer models ===============================================
    path = SEMEVAL_FEATS_TRAIN / '{task}_onlytext.pkl'.format(task=SEMEVAL_TASKA)
    if not path.exists():
        with open(path, 'wb') as file:
            cleaned_tweet = np.asarray([handle_twitter_specific_tags(text)for text in X_stance_train])
            pickle.dump(cleaned_tweet, file)

    path = SEMEVAL_FEATS_TRAIN / '{task}_onlytext.pkl'.format(task=SEMEVAL_TASKB)
    if not path.exists():
        with open(path, 'wb') as file:
            cleaned_tweet = np.asarray([handle_twitter_specific_tags(text)for text in X_veracity_train])
            pickle.dump(cleaned_tweet, file)

    path = SEMEVAL_FEATS_DEV / '{task}_onlytext.pkl'.format(task=SEMEVAL_TASKA)
    if not path.exists():
        with open(path, 'wb') as file:
            cleaned_tweet = np.asarray([handle_twitter_specific_tags(text)for text in X_stance_dev])
            pickle.dump(cleaned_tweet, file)

    path = SEMEVAL_FEATS_DEV / '{task}_onlytext.pkl'.format(task=SEMEVAL_TASKB)
    if not path.exists():
        with open(path, 'wb') as file:
            cleaned_tweet = np.asarray([handle_twitter_specific_tags(text)for text in X_veracity_dev])
            pickle.dump(cleaned_tweet, file)

    path = SEMEVAL_FEATS_TEST / '{task}_onlytext.pkl'.format(task=SEMEVAL_TASKA)
    if not path.exists():
        with open(path, 'wb') as file:
            cleaned_tweet = np.asarray([handle_twitter_specific_tags(text)for text in X_stance_test])
            pickle.dump(cleaned_tweet, file)

    path = SEMEVAL_FEATS_TEST / '{task}_onlytext.pkl'.format(task=SEMEVAL_TASKB)
    if not path.exists():
        with open(path, 'wb') as file:
            cleaned_tweet = np.asarray([handle_twitter_specific_tags(text)for text in X_veracity_test])
            pickle.dump(cleaned_tweet, file)

    # ======================== Save the labels ===============================================
    path = SEMEVAL_FEATS_TRAIN / '{task}_label.pkl'.format(task=SEMEVAL_TASKA)
    if not path.exists():
        with open(path, 'wb') as file:
            pickle.dump(y_stance_train, file)

    path = SEMEVAL_FEATS_TRAIN / '{task}_label.pkl'.format(task=SEMEVAL_TASKB)
    if not path.exists():
        with open(path, 'wb') as file:
            pickle.dump(y_veracity_train, file)

    path = SEMEVAL_FEATS_DEV / '{task}_label.pkl'.format(task=SEMEVAL_TASKA)
    if not path.exists():
        with open(path, 'wb') as file:
            pickle.dump(y_stance_dev, file)

    path = SEMEVAL_FEATS_DEV / '{task}_label.pkl'.format(task=SEMEVAL_TASKB)
    if not path.exists():
        with open(path, 'wb') as file:
            pickle.dump(y_veracity_dev, file)

    path = SEMEVAL_FEATS_TEST / '{task}_label.pkl'.format(task=SEMEVAL_TASKA)
    if not path.exists():
        with open(path, 'wb') as file:
            pickle.dump(y_stance_test, file)

    path = SEMEVAL_FEATS_TEST / '{task}_label.pkl'.format(task=SEMEVAL_TASKB)
    if not path.exists():
        with open(path, 'wb') as file:
            pickle.dump(y_veracity_test, file)

    # ======================== Save the features ===============================================
    for feature, fnc in FEATURES_MAP.items():
        logger.info('Feature {feature} extraction for {data_set}', feature=feature, data_set='train')
        path = SEMEVAL_FEATS_TRAIN / '{task}_{feature}.pkl'.format(task=SEMEVAL_TASKA, feature=feature)
        if not path.exists():
            with open(path, 'wb') as file:
                feats = np.asarray(
                    [[value for value in fnc(text).values()] for text in X_stance_train])
                pickle.dump(feats, file)
        else:
            logger.info('Feature {feature} for {data_set} already exists.', feature=feature, data_set='train')

        path = SEMEVAL_FEATS_TRAIN / '{task}_{feature}.pkl'.format(task=SEMEVAL_TASKB, feature=feature)
        if not path.exists():
            with open(path, 'wb') as file:
                feats = np.asarray(
                    [[value for value in fnc(text).values()] for text in X_veracity_train])
                pickle.dump(feats, file)
        else:
            logger.info('Feature {feature} for {data_set} already exists.', feature=feature, data_set='train')

        logger.info('Feature {feature} extraction for {data_set}', feature=feature, data_set='dev')
        path = SEMEVAL_FEATS_DEV / '{task}_{feature}.pkl'.format(task=SEMEVAL_TASKA, feature=feature)
        if not path.exists():
            with open(path, 'wb') as file:
                feats = np.asarray(
                    [[value for value in fnc(text).values()] for text in X_stance_dev])
                pickle.dump(feats, file)
        else:
            logger.info('Feature {feature} for {data_set} already exists.', feature=feature, data_set='dev')

        path = SEMEVAL_FEATS_DEV / '{task}_{feature}.pkl'.format(task=SEMEVAL_TASKB, feature=feature)
        if not path.exists():
            with open(path, 'wb') as file:
                feats = np.asarray(
                    [[value for value in fnc(text).values()] for text in X_veracity_dev])
                pickle.dump(feats, file)
        else:
            logger.info('Feature {feature} for {data_set} already exists.', feature=feature, data_set='dev')

        logger.info('Feature {feature} extraction for {data_set}', feature=feature, data_set='test')
        path = SEMEVAL_FEATS_TEST / '{task}_{feature}.pkl'.format(task=SEMEVAL_TASKA, feature=feature)
        if not path.exists():
            with open(path, 'wb') as file:
                feats = np.asarray(
                    [[value for value in fnc(text).values()] for text in X_stance_test])
                pickle.dump(feats, file)
        else:
            logger.info('Feature {feature} for {data_set} already exists.', feature=feature, data_set='test')

        path = SEMEVAL_FEATS_TEST / '{task}_{feature}.pkl'.format(task=SEMEVAL_TASKB, feature=feature)
        if not path.exists():
            with open(path, 'wb') as file:
                feats = np.asarray(
                    [[value for value in fnc(text).values()] for text in X_veracity_test])
                pickle.dump(feats, file)
        else:
            logger.info('Feature {feature} for {data_set} already exists.', feature=feature, data_set='test')


def get_features(feature, task, collection_type):
    '''
    :param feature: e.g complexity, bias
    :type feature: str
    :param task: e.g subtaskaenglish, subtaskbenglish
    :type task: str
    :param collection_type: e.g train, dev, test
    :type collection_type: str
    :return: features
    :rtype:
    '''
    # assert feature not in FEATURES_MAP.keys(), '{} is not registered'.format(feature)

    feature_path = None
    if collection_type == 'train':
        feature_path = SEMEVAL_FEATS_TRAIN
    elif collection_type == 'dev':
        feature_path = SEMEVAL_FEATS_DEV
    elif collection_type == 'test':
        feature_path = SEMEVAL_FEATS_TEST
    else:
        raise Exception('Not valid task {}'.format(task))

    feature_path = feature_path / '{task}_{feature}.pkl'.format(task=task, feature=feature)

    # assert not os.path.exists(feature_path), 'This path does not exist {}'.format(feature_path)

    with open(feature_path, 'rb'):
        file = open(feature_path, 'rb')
        return pickle.load(file)


# -*- coding: utf-8 -*-
from test_tube import HyperOptArgumentParser

from coinform_content_analysis.data_loader.data_utils import get_features


def collate_lists(text: list, label: list) -> dict:
    """ Converts each line into a dictionary. """
    collated_dataset = []
    for i in range(len(text)):
        collated_dataset.append({"text": str(text[i]), "label": str(label[i])})
    return collated_dataset


def rumoureval_stance_dataset(
    hparams: HyperOptArgumentParser, train=True, val=True, test=True
):
    X_train_stance = get_features('onlytext', 'subtaskaenglish', collection_type='train')
    y_train_stance = get_features('label', 'subtaskaenglish', collection_type='train')
    assert len(X_train_stance) == len(y_train_stance)

    X_dev_stance = get_features('onlytext', 'subtaskaenglish', collection_type='dev')
    y_dev_stance = get_features('label', 'subtaskaenglish', collection_type='dev')
    assert len(X_dev_stance) == len(y_dev_stance)

    X_test_stance = get_features('onlytext', 'subtaskaenglish', collection_type='test')
    y_test_stance = get_features('label', 'subtaskaenglish', collection_type='test')

    assert len(X_test_stance) == len(y_test_stance)

    logger.info('Train sample size {sample} label {label}'.format(sample=len(X_train_stance), label=len(y_train_stance)))
    logger.info('Dev sample size {sample} label {label}'.format(sample=len(X_dev_stance), label=len(y_dev_stance)))
    logger.info('Test sample size {sample} label {label}'.format(sample=len(X_test_stance), label=len(y_test_stance)))


    func_out = []
    if train:
        func_out.append(Dataset(collate_lists(X_train_stance, y_train_stance)))
    if val:
        func_out.append(Dataset(collate_lists(X_dev_stance, y_dev_stance)))
    if test:
        func_out.append(Dataset(collate_lists(X_test_stance, y_test_stance)))

    return tuple(func_out)


def rumoureval_veracity_dataset(
    hparams: HyperOptArgumentParser, train=True, val=True, test=True
):
    X_train_veracity = get_features('onlytext', 'subtaskbenglish', collection_type='train')
    y_train_veracity = get_features('label', 'subtaskbenglish', collection_type='train')

    assert len(X_train_veracity) == len(y_train_veracity)

    X_dev_veracity = get_features('onlytext', 'subtaskbenglish', collection_type='train')
    y_dev_veracity = get_features('label', 'subtaskbenglish', collection_type='train')

    assert len(X_dev_veracity) == len(y_dev_veracity)

    X_test_veracity = get_features('onlytext', 'subtaskbenglish', collection_type='train')
    y_test_veracity = get_features('label', 'subtaskbenglish', collection_type='train')

    assert len(X_test_veracity) == len(y_test_veracity)

    logger.info('Train sample size {sample} label {label}'.format(sample=len(X_train_veracity), label=y_train_veracity))
    logger.info('Dev sample size {sample} label {label}'.format(sample=len(X_dev_veracity), label=y_dev_veracity))
    logger.info('Test sample size {sample} label {label}'.format(sample=len(X_test_veracity), label=y_test_veracity))

    func_out = []
    if train:
        func_out.append(Dataset(collate_lists(X_train_veracity, y_train_veracity)))
    if val:
        func_out.append(Dataset(collate_lists(X_dev_veracity, y_dev_veracity)))
    if test:
        func_out.append(Dataset(collate_lists(X_test_veracity, y_test_veracity)))

    return tuple(func_out)


if __name__ == '__main__':
    extract_rumoureval2019_features()
