#!/bin/bash

mkdir stats
mkdir stats/test
mkdir log_files
mkdir log_files/test
mkdir log_files/test/maddpg
mkdir log_files/test/m3ddpg
mkdir log_files/test/maddpg_d
mkdir log_files/test/m3ddpg_d

for i in 1 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
do
    python test_maddpg.py log_files/test/maddpg/moving_avg_$i.txt models/maddpg/ $i 0 > log_files/test/maddpg/training_log_$i.txt &
    python test_maddpg.py log_files/test/maddpg_d/moving_avg_$i.txt models/maddpg_d/ $i 1 > log_files/test/maddpg_d/training_log_$i.txt &
done
