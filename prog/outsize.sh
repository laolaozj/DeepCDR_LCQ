#!/bin/bash

echo "drop out"
#python run_DeepCDR2.py -unit_list 128 128 128 && echo "Launched" &
#P=$!
#echo $P
#wait $P
#python run_DeepCDR2.py -unit_list 200 200 200 && echo "Launched" &
#P=$!
#echo $P
#wait $P
#python run_DeepCDR2.py -unit_list 300 300 300 && echo "Launched" &
#P=$!
#echo $P
#wait $P
#python run_DeepCDR2.py -unit_list 400 400 400 && echo "Launched" &
#P=$!-
#echo $P
#wait $P
python run_DeepCDR2.py && echo "Launched" &
P=$!
echo $P
wait $P
python run_DeepCDR2.py && echo "Launched" &
P=$!
echo $P
wait $P
python run_DeepCDR2.py && echo "Launched" &
P=$!
echo $P
wait $P

