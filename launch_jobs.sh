#!/bin/bash

for i in {0..75}        # Number of parallel processes
do
   start=$((i * 2))     # System_id job i starts processing
   end=$(((i+1) * 2))   # System_id job i stops processing
   nohup python resonance_chain_planetary_system.py $start $end >logfiles/out${i}.log &
done