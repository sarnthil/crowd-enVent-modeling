{
    "dataset_reader": {
        "type": "tsv",
        "y_column": "pleasantness",
        "scaling": true,
    },
    "train_data_path": "../sources/crowd-enVent-train.tsv",
    "validation_data_path": "../sources/crowd-enVent-val.tsv",
    "test_data_path": "../sources/crowd-enVent-test.tsv",
    "model": {
        "type": "crowd_regressor",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10,
                },
            }
        },
        "seq2vec_encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 10,
        },
        "dropout": 0.1
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "num_epochs": 10,
        "patience": 32,
        "grad_norm": 4.0,
        "validation_metric": "+PearsonCorrelation",
        "optimizer": {
            "type": "adam",
            "lr": 3e-5
        }
    },
    "random_seed": std.parseInt(std.extVar("RANDOM_SEED")),
    "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
    "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
}
