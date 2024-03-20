# Copyright 2023 Iguazio
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
from huggingface_dpo_trainer import dpo_train

import mlrun

def test_dpo_fn():
    ctx = mlrun.get_or_create_ctx(name='test_dpo')
    train_dataset = "unalignment/toxic-dpo-v0.2"
    training_arguments = {
            "evaluation_strategy": "steps",
            "do_eval": True,
            "optim": "paged_adamw_8bit",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "per_device_eval_batch_size": 1,
            "log_level": "info",
            "save_steps": 100,
            "learning_rate": 5e-7,
            "eval_steps": 100,
            "num_train_epochs": 1,
            "max_steps": 100,
            "warmup_steps": 20,
            "fp16": True,
            "lr_scheduler_type": "cosine",
            "remove_unused_columns": True,
            "gradient_checkpointing": True,
        }
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = model_name
    dpo_train(
           context = ctx,
           train_dataset = train_dataset,
           model = (model_name,"transformers.AutoModelForCausalLM"),
           tokenizer = tokenizer,
           dataset_columns_to_train = ['chosen', 'rejected'],
           training_config = training_arguments,
           use_cuda = True,
           beta = 0.1,
           split='train',
           )

    

def test_dpo_train():

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = model_name
    dop_trainer = mlrun.import_function("function.yaml")

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
        "ref_model": None,
        "tokenizer": tokenizer,
        "train_dataset": "Abirate/english_quotes",
        "training_config": training_arguments,
        "dataset_columns_to_train": "quote",
        "model_pretrained_config": {"use_cache": False},

        "use_cuda": False,
    }

    try:
        with tempfile.TemporaryDirectory() as test_directory:
            dpo_trainer.run(
                local=True,
                params=params,
                handler="dpo_train",
                returns=["model"],
                workdir=test_directory,
            )

    except Exception as exception:
        print(f"- The training failed - raised the following error:\n- {exception}")
