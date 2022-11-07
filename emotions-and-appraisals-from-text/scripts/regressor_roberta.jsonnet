local transformer_model = "roberta-large";
local epochs = 1;
local batch_size = 1;
local num_gradient_accumulation_steps = 1;
local training_data_size = 4200;
{

    "dataset_reader": {
        "type": "tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model,
                "max_length": 128
            },
        },
        "max_tokens": 128,
        "y_column": std.extVar("Y_COLUMN"),
    },
    "train_data_path": "../sources/crowd-enVent-train.tsv",
    "validation_data_path": "../sources/crowd-enVent-val.tsv",
    "test_data_path": "../sources/crowd-enVent-test.tsv",

    "model": {
        "type": "crowd_regressor_roberta",
        "hidden_size": 128,
        "train_base": false
    },
    "data_loader": {
        "type": "simple",
        "batch_size": 16,
        "shuffle": true,
    },
    "trainer": {
        "learning_rate_scheduler": {
          "type": "slanted_triangular",
          "num_epochs": 10,
          "num_steps_per_epoch": 100,
          "cut_frac": 0.06
        },
        "num_epochs": 1,
        "patience": 4,
        "grad_norm": 4.0,
        "validation_metric": "+PearsonCorrelation",
        "cuda_device": -1,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3.0e-5,
            "weight_decay": 0.1,
        },
    },
    "random_seed": std.parseInt(std.extVar("RANDOM_SEED")),
    "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
    "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
}
