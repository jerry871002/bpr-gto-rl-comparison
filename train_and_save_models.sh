#!/bin/bash

mkdir models
mkdir models/maddpg
mkdir models/m3ddpg
mkdir models/maddpg_d
mkdir models/m3ddpg_d

mkdir stats

mkdir log_files
mkdir log_files/maddpg
mkdir log_files/m3ddpg
mkdir log_files/maddpg_d
mkdir log_files/m3ddpg_d

python train_maddpg.py log_files/maddpg/moving_avg.txt > log_files/maddpg/training_log.txt &
python train_m3ddpg.py v0 log_files/m3ddpg/moving_avg.txt > log_files/m3ddpg/training_log.txt &
python train_maddpg_deterministic.py log_files/maddpg_d/moving_avg.txt > log_files/maddpg_d/training_log.txt &
python train_m3ddpg_deterministic.py v0 log_files/m3ddpg_d/moving_avg.txt > log_files/m3ddpg_d/training_log.txt &
