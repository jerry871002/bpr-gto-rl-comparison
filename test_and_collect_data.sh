#!/bin/bash

mkdir stats
mkdir stats/test
mkdir log_files
mkdir log_files/test
mkdir log_files/test/maddpg
# mkdir log_files/test/m3ddpg
mkdir log_files/test/maddpg_d
# mkdir log_files/test/m3ddpg_d

for i in 1 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000
do
    python test_maddpg.py log_files/test/maddpg/moving_avg_$i.txt models/maddpg/ $i 0 > log_files/test/maddpg/training_log_$i.txt &
    python test_maddpg.py log_files/test/maddpg_d/moving_avg_$i.txt models/maddpg_d/ $i 1 > log_files/test/maddpg_d/training_log_$i.txt &
    # python test_m3ddpg.py log_files/test/m3ddpg/moving_avg_$i.txt models/m3ddpg/ $i 0 > log_files/test/m3ddpg/training_log_$i.txt &
    # python test_m3ddpg.py log_files/test/m3ddpg_d/moving_avg_$i.txt models/m3ddpg_d/ $i 1 > log_files/test/m3ddpg_d/training_log_$i.txt &
done
