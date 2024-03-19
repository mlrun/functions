# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile

import mlrun


def test_dpo_train():

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
        "train_dataset": "HuggingFaceH4/orca_dpo_pairs",
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
