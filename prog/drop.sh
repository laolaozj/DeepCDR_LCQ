#!/bin/bash

echo "drop out"
python run_DeepCDR2.py -Dropout_rate=0.2 && echo "Launched" &
P=$!
echo $P
wait $P
python run_DeepCDR2.py -Dropout_rate=0.2 && echo "Launched" &
P=$!
echo $P
wait $P
python run_DeepCDR2.py -Dropout_rate=0.2 && echo "Launched" &
P=$!
echo $P
wait $P


