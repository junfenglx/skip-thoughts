#!/bin/bash

echo "test on RTE1"
python ./decoded_rte_mlp.py 1 > decoded_rte1_mlp.log

echo "test on RTE2"
python ./decoded_rte_mlp.py 2 > decoded_rte2_mlp.log

echo "test on RTE3"
python ./decoded_rte_mlp.py 3 > decoded_rte3_mlp.log


echo "do train rte_on_skipthoughts models"

./rte_on_skipthoughts_run.sh
