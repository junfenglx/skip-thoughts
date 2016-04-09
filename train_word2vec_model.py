# -*- coding: utf-8 -*-

# Created by junfeng on 3/28/16.

# logging config
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import os.path
import sys
import multiprocessing

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


class Characterize(object):

    def __init__(self, inp):
        self.inp = inp
        self.line_sentence = LineSentence(inp, max_sentence_length=100000)

    def __iter__(self):
        for article in self.line_sentence:
            # print(article)
            article = ''.join(article)
            article = [c for c in article]
            yield article


if __name__ == '__main__':

    logger.info('Running %s' % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        print(sys.argv[0] + ' wiki.text output_vector_model.bin output_vector_model.text')
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]

    model = Word2Vec(Characterize(inp), size=1000, window=5, min_count=1,
                     sample=1e-5, hs=0, negative=5, iter=10,
                     workers=int(multiprocessing.cpu_count() / 4)
                     )

    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    model.save(outp1)
    model.save_word2vec_format(outp2, binary=False)
