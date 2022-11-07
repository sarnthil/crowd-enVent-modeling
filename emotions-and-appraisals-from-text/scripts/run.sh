#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
# source $HOME/venvs/allennlp/bin/activate
# cd $HOME/allennlp-guide/scripts

# cd scripts

for seed in 3141592654
do
    export RANDOM_SEED="$seed"
    export NUMPY_SEED="${RANDOM_SEED}"
    export PYTORCH_SEED="${RANDOM_SEED}"
    export Y_COLUMN="$1"
    if [ "$2" = "classification" ]
    then
        export JSONNET="classifier_roberta.jsonnet"
    elif [ "$2" = "regression" ]
    then
        export JSONNET="regressor_roberta.jsonnet"
    elif [ "$2" = "combined" ]
    then
        export JSONNET="classifier_roberta_with_appraisals.jsonnet"
    else
        echo "Unknown setting $2"
        exit 1
    fi
    if [ "$Y_COLUMN" = "emotion" ]
    then
        export DO_BINNING=0
    else
    export DO_BINNING=1
    fi

    time allennlp train "$JSONNET" -s "../workdata/$2-${Y_COLUMN}-model-roberta-${RANDOM_SEED}" --include-package crowdenvent
done
