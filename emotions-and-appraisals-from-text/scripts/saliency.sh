#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
source $HOME/venvs/allennlp/bin/activate
cd $HOME/allennlp-guide/scripts
time python saliency_maps.py --no-interpret --data ../workdata/crowd-enVent_validation_deduplicated.tsv $1
