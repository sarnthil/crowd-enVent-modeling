#!/bin/bash

# for model in ../workdata/combined-emotion-model-roberta-23 ../workdata/classification-*
for model in ../workdata/regression-*
do
	if ! [ -d "$model" ]
	then
		continue
	fi
	if ! [ -f "$model/best.th" ]
	then
		continue
	fi
	sbatch -p dev_gpu_4 -J "sal-$model" -- saliency.sh "$model"
done
