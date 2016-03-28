# -*- coding: utf-8 -*-

# Created by junfeng on 3/28/16.

# logging config
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

import skipthoughts
model = skipthoughts.load_model()

X = [
    'Hello, skip thoughts',
]
vectors = skipthoughts.encode(model, X)
print(vectors)