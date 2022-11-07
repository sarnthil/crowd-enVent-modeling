#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
source $HOME/venvs/allennlp/bin/activate
cd $HOME/allennlp-guide/scripts

time python lime-analysis.py -s 859 -c ../sources/crowd-enVent_validation_deduplicated.tsv ../workdata/classification-emotion-model-roberta-3141/
