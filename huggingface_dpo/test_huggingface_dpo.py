import tempfile

import mlrun


def test_train():

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = model_name
    auto_trainer = mlrun.import_function("function.yaml")

    training_arguments = {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 2,
        "max_steps": 10,
        "learning_rate": 2e-4,
        "logging_steps": 1,
    }

    params = {
        "model": (model_name, "transformers.AutoModelForCausalLM"),
        "tokenizer": tokenizer,
        "train_dataset": "Abirate/english_quotes",
        "training_config": training_arguments,
        "dataset_columns_to_train": "quote",
        "model_pretrained_config": {"use_cache": False},
        "use_cuda": False,
    }

    try:
        with tempfile.TemporaryDirectory() as test_directory:
            auto_trainer.run(
                local=True,
                params=params,
                handler="dpo_train",
                returns=["model"],
                workdir=test_directory,
            )

    except Exception as exception:
        print(f"- The training failed - raised the following error:\n- {exception}")
