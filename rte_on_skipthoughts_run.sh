#!/bin/bash

echo "test on RTE1"
python rte_on_skipthoughts.py 1 > pre-train_encoder_epoch20_rte1.log

echo "test on RTE2"
python rte_on_skipthoughts.py 2 > pre-train_encoder_epoch20_rte2.log

echo "test on RTE3"
python rte_on_skipthoughts.py 3 > pre-train_encoder_epoch20_rte3.log
