# -*- coding: utf-8 -*-

# Created by junfeng on 4/5/16.

# logging config
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    import cPickle as pickle
except ImportError as e:
    import pickle

from sklearn.externals import joblib
import bs4

import numpy as np

import skipthoughts


def read_model():
    model = skipthoughts.load_model()
    return model


def read_rte_xml():
    filename = './data/rte-dataset.xml'
    with open(filename, 'r') as f:
        content = f.read()
        soup = bs4.BeautifulSoup(content, 'xml')
        pairs = soup.find_all('pair')
        sample_length = len(pairs)
        logger.info('sample length: {0}'.format(sample_length))
        ts = []
        hs = []
        labels = np.zeros(sample_length, dtype=int)
        samples = []
        for i, pair in enumerate(pairs):
            value = pair.get('value')
            if value == 'TRUE':
                labels[i] = 1
            t = pair.find('t')
            h = pair.find('h')
            t = t.string.strip()
            h = h.string.strip()
            ts.append(t)
            hs.append(h)
            samples.append(u'{0} {1}'.format(t, h))
            if i % 1000 == 0:
                logger.info('processed sample {0}'.format(i))
        logger.info('unique ts: {0}, unique hs: {1}'.format(len(set(ts)), len(set(hs))))
        logger.info('unique sample: {0}'.format(len(set(samples))))
        logger.info('TRUE labels: {0}'.format(np.sum(labels)))
        return ts, hs, labels


if __name__ == '__main__':
    logger.info('read rte dataset xml file ...')
    ts, hs, labels = read_rte_xml()
    logger.info('read model ...')
    model = read_model()
    logger.info('encoding ts ...')
    vectorized_ts = skipthoughts.encode(model, ts)
    logger.info('encoding hs ...')
    vectorized_hs = skipthoughts.encode(model, hs)
    logger.info('dump to file ...')
    joblib.dump((vectorized_ts, vectorized_hs, labels), './data/processed-rte-dataset.pkl')
    logger.info('done')
