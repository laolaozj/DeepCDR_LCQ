#!/bin/bash

#echo "acti"
#python run_DeepCDR3.py -activation "elu" && echo "Launched" &
#P=$!
#echo $P
#wait $P
#python run_DeepCDR3.py -activation "softplus" && echo "Launched" &
#P=$!
#echo $P
#wait $P
#python run_DeepCDR3.py -activation "tanh" && echo "Launched" &
#P=$!
#echo $P
#wait $P
#python run_DeepCDR3.py -activation "sigmoid" && echo "Launched" &
#P=$!
#echo $P
#wait $P
echo "drop out"
python run_DeepCDR3.py && echo "Launched" &
P=$!
echo $P
wait $P
python run_DeepCDR3.py && echo "Launched" &
P=$!
echo $P
wait $P
python run_DeepCDR3.py && echo "Launched" &
P=$!
echo $P
wait $P



