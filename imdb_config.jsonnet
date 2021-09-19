{
    "dataset_reader": {
        "type": "imdb2",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "max_length": 512,
                "model_name": "roberta-base"
            }
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "max_length": 512,
            "model_name": "roberta-base"
        }
    },
    "model": {
        "type": "basic_classifier",
        "namespace": "tags",
        "seq2vec_encoder": {
            "type": "bert_pooler",
            "dropout": 0.1,
            "pretrained_model": "roberta-base"
        },
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "max_length": 512,
                    "model_name": "roberta-base"
                }
            }
        }
    },
    "train_data_path": "train",
    "test_data_path": "test",
    "trainer": {
        "num_epochs": 5,
	 "learning_rate_scheduler": {
	    "type": "slanted_triangular",
	    "cut_frac": 0.06		
	},
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 2e-05,
            "weight_decay": 0.1
        }
    },
    "evaluate_on_test": true,
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 8,
            "sorting_keys": [
                "tokens"
            ]
        }
    }
}