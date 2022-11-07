local transformer_model = "roberta-large";
local transformer_dim = 1024;

{
    "dataset_reader" : {
        "type": "tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": transformer_model,
                "max_length": 128
            }
        },
        "max_tokens": 128,
        "y_column": std.extVar("Y_COLUMN"),
        "binning": std.parseInt(std.extVar("DO_BINNING")) != 0,
    },
    "train_data_path": "../sources/crowd-enVent-train.tsv",
    "validation_data_path": "../sources/crowd-enVent-val.tsv",
    "test_data_path": "../sources/crowd-enVent-test.tsv",
    "model": {
        "type": "crowd_classifier",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": transformer_model
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": transformer_model,
        }
    },
    "data_loader": {
        "batch_size": 16,
        "shuffle": true
    },
    "trainer": {
        "learning_rate_scheduler": {
          "type": "slanted_triangular",
          "num_epochs": 10,
          "num_steps_per_epoch": 100,
          "cut_frac": 0.06
        },
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3.0e-5,
            "weight_decay": 0.1,
        },
        "num_epochs": 20
    },
    "random_seed": std.parseInt(std.extVar("RANDOM_SEED")),
    "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
    "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
}
