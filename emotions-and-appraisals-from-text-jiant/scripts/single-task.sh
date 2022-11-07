#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=1440

source $HOME/venvs/jiant/bin/activate
timestamp=$(date -Iseconds)
cd $HOME/jiant
mkdir logs
time python crowdenvent_single.py $1 >$HOME/jiant/logs/stdout-$1-$timestamp.log 2>$HOME/jiant/logs/stderr-$1-$timestamp.log
