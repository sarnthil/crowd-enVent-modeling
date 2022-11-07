#!/bin/bash

sbatch -t 06:00:00 -p gpu_4 -J "lime-$model" -- scripts/lime-analysis.py -c sources/crowd-enVent_validation_deduplicated.tsv workdata/classification-emotion-model-roberta-3141/
