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
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = model_name

    ctx = mlrun.get_or_create_ctx(name="test_dpo")
    train_dataset = "unalignment/toxic-dpo-v0.2"
    eval_dataset = "unalignment/toxic-dpo-v0.2"
    training_arguments = {
        "evaluation_strategy": "steps",
        "do_eval": False,
        "optim": "paged_adamw_8bit",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "per_device_eval_batch_size": 1,
        "log_level": "info",
        "save_steps": 1,
        "learning_rate": 5e-7,
        "eval_steps": 1,
        "num_train_epochs": 1,
        "max_steps": 1,
        "warmup_steps": 1,
        "fp16": True,
        "lr_scheduler_type": "cosine",
        "remove_unused_columns": True,
        "gradient_checkpointing": True,
    }
    dpo_train(
        context=ctx,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=True,
        model=model_name,
        tokenizer=tokenizer,
        training_config=training_arguments,
        use_cuda=True,
        beta=0.1,
    )
