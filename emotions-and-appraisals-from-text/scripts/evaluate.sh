#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --partition=dev_gpu_4
# source $HOME/venvs/allennlp/bin/activate
# cd $HOME/allennlp-guide/scripts

cd scripts

export seed="345"
# export type="$2"
# export appraisals="$3"
# for seed in ....
for run in emotion suddenness familiarity predict_event pleasantness unpleasantness goal_relevance chance_responsblt self_responsblt other_responsblt predict_conseq goal_support urgency self_control other_control chance_control accept_conseq standards social_norms attention not_consider effort
do
    for type in classification combined regression
    do
        time allennlp predict ../workdata/$type-$run-model-roberta-$seed ../sources/crowd-enVent_validation_deduplicated.tsv --use-dataset-reader --output-file ../workdata/$type-$run-model-roberta-$seed/predictions.json  --include-package crowdenvent --predictor crowd_classifier
        python evaluate_preds.py --format json --model ../workdata/$type-$run-model-roberta-$seed
        time allennlp predict ../workdata/$type-$run-model-roberta-$seed ../sources/crowd-enVent_predicted_gold.tsv --use-dataset-reader --output-file ../workdata/$type-$run-model-roberta-$seed/predictions_silver.json  --include-package crowdenvent --predictor crowd_classifier
        python evaluate_preds.py --format json-silver --model ../workdata/$type-$run-model-roberta-$seed
    done
done

cd ..
# if [ $appraisals = "gold" ]
# then
# 	time allennlp predict ../workdata/$type-emotion-model-roberta-$seed/ ../sources/crowd-enVent_validation_deduplicated.tsv --use-dataset-reader --output-file ../workdata/$type-emotion-model-roberta-$seed/predictions.json  --include-package crowdenvent --predictor crowd_classifier
# else
# 	time allennlp predict ../workdata/$type-emotion-model-roberta-$seed/ ../sources/crowd-enVent_predicted_gold.tsv --use-dataset-reader --output-file ../workdata/$type-emotion-model-roberta-$seed/predictions_silver.json  --include-package crowdenvent --predictor crowd_classifier
# fi
