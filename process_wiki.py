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

from gensim.corpora import WikiCorpus

if __name__ == '__main__':

    logger.info("Running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print(sys.argv[0] + ' wiki-pages-articles.xml.bz2 output_text_name')
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = ' '
    i = 0

    with open(outp, 'w') as output:
        wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
        for text in wiki.get_texts():
            output.write(space.join(text) + '\n')
            i += 1
            if (i % 10000 == 0):
                logger.info('Saved ' + str(i) + ' articles')

        logger.info('Finished saved ' + str(i) + ' articles')
