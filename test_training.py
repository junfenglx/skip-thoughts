# -*- coding: utf-8 -*-

# Created by junfeng on 3/29/16.

# logging config
import codecs
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

from training import vocab
from training import train as st

WIKI_ZH_SIM_PATH = './zh_data/wiki_zh_sim_2016-03-28.text'
DICTIONARY_LOC = './zh_data/wiki_zh_sim_2016-03-28_dictionary.pkl'


def characterize(inp, maxline=None):

    with codecs.open(inp, 'rU', 'utf-8') as f:
        i = 0
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split()
            line = ''.join(words)
            yield ' '.join(line)
            i += 1
            if i % 10000 == 0:
                logger.info('processed {0} lines'.format(i))
            if maxline is not None:
                if i >= maxline:
                    return


def build_vocab():
    logger.info('Build dictionary ...')
    worddict, wordcount = vocab.build_dictionary(characterize(WIKI_ZH_SIM_PATH))
    logger.info('vocab size: {0}'.format(len(worddict)))
    logger.info('Save dictionary ...')
    vocab.save_dictionary(worddict, wordcount, DICTIONARY_LOC)
    logger.info('Done')

def train():
    logger.info('Read sentences ...')
    X = list(characterize(WIKI_ZH_SIM_PATH, maxline=128))
    logger.info('Train ...')
    st.trainer(X, dim_word=10, dim=20, max_epochs=1, n_words=100, maxlen_w=7000, saveto='./zh_data/model_toy.npz', dictionary=DICTIONARY_LOC)
    logger.info('Done')

if __name__ == '__main__':
    # build_vocab()
    train()
