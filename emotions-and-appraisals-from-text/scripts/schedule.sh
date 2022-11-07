#!/bin/bash
cd scripts

# took these columns directly from the tsv, in the order they appear
for run in emotion suddenness familiarity predict_event pleasantness unpleasantness goal_relevance chance_responsblt self_responsblt other_responsblt predict_conseq goal_support urgency self_control other_control chance_control accept_conseq standards social_norms attention not_consider effort
do
    for setting in classification combined regression
    do
        if [ "$setting" = "regression" ] && [ "$run" = "emotion" ]
        then
            # we can't do emotion regression
            continue
        fi
        if which sbatch >/dev/null
        then
            sbatch -p dev_gpu_4 -t 25 -J "A-$run" run.sh "$run" "$setting"
        elif which qsub >/dev/null
        then
            qsub -N "A-$run" run.sh "$run" "$setting"
        else
            run.sh "$run" "$setting"
        fi
    done
done
