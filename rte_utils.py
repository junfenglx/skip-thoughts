# -*- coding: utf-8 -*-

# Created by junfeng on 4/12/16.

import os.path

# logging config
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from nltk.corpus import rte


def get_sentence_sample(pairs):
    sample_length = len(pairs)
    logger.info('sample length: {0}'.format(sample_length))
    ts = []
    hs = []
    labels = np.empty(sample_length, dtype=int)
    samples = []
    for i, pair in enumerate(pairs):
        value = pair.value
        labels[i] = value
        t = pair.text
        h = pair.hyp
        ts.append(t)
        hs.append(h)
        samples.append(u'{0} {1}'.format(t, h))
        if i % 1000 == 0:
            logger.info('processed sample {0}'.format(i))
    logger.info('unique ts: {0}, unique hs: {1}'.format(len(set(ts)), len(set(hs))))
    logger.info('unique sample: {0}'.format(len(set(samples))))
    logger.info('TRUE labels: {0}'.format(np.sum(labels)))
    return ts, hs, labels


def read_rte_from_nltk(version=3):
    train_saved_path = './data/raw-rte{0}-train.csv'.format(version)
    test_saved_path = './data/raw-rte{0}-test.csv'.format(version)
    if os.path.isfile(train_saved_path) and os.path.isfile(test_saved_path):
        rte_train = pd.read_csv(train_saved_path)
        rte_test = pd.read_csv(test_saved_path)
        return RTEData(rte_train, rte_test)

    train_xml = 'rte{0}_dev.xml'.format(version)
    test_xml = 'rte{0}_test.xml'.format(version)
    train_pairs = rte.pairs(train_xml)
    test_pairs = rte.pairs(test_xml)
    train_ts, train_hs, train_labels = get_sentence_sample(train_pairs)
    test_ts, test_hs, test_labels = get_sentence_sample(test_pairs)
    rte_train = pd.DataFrame(
            data=dict(text=train_ts, hypothesis=train_hs, label=train_labels)
    )
    rte_test = pd.DataFrame(
            data=dict(text=test_ts, hypothesis=test_hs, label=test_labels)
    )
    rte_train.to_csv(train_saved_path, index=False, encoding='utf-8')
    rte_test.to_csv(test_saved_path, index=False, encoding='utf-8')
    return RTEData(rte_train, rte_test)


class RTEData(object):
    def __init__(self, rte_train_df, rte_test_df):
        self.train_df = rte_train_df
        self.test_df = rte_test_df


if __name__ == '__main__':
    rte_data = read_rte_from_nltk(version=1)
    print(rte_data.train_df[:10])
    rte_data = read_rte_from_nltk(version=2)
    rte_data = read_rte_from_nltk(version=3)
    logger.info('done')
