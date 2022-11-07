#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=10
#SBATCH --job-name=test

source $HOME/venvs/jiant/bin/activate
timestamp=$(date -Iseconds)
cd $HOME/jiant
mkdir logs
time python crowdenvent_single.py emo_cls >$HOME/jiant/logs/stdout-$timestamp.log 2>$HOME/jiant/logs/stderr-$timestamp.log
