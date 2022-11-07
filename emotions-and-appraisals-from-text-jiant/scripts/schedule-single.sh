#!/bin/bash

sbatch -p gpu_4 -t 20 -J emo_cls scripts/single-task.sh emo_cls

for task in suddenness familiarity predict_event pleasantness unpleasantness goal_relevance chance_responsblt self_responsblt other_responsblt predict_conseq goal_support urgency self_control other_control chance_control accept_conseq standards social_norms attention not_consider effort
do
    for setting in cls reg
    do
        sbatch -p gpu_4 -t 20 -J "${task}_${setting}" scripts/single-task.sh "${task}_${setting}"
    done
done
