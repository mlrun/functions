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

import importlib
import os
import shutil
import tempfile
import zipfile
from abc import ABC
from typing import Dict, List, Tuple, Union

import mlrun
import numpy as np
import pandas as pd
import peft
import torch
import transformers
from datasets import Dataset, load_dataset
from mlrun.artifacts.manager import Artifact, PlotlyArtifact
from mlrun.datastore import is_store_uri
from mlrun.frameworks._common import CommonTypes, MLRunInterface
from mlrun.utils import logger
from trl import DPOTrainer
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from plotly import graph_objects as go
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          PreTrainedModel, PreTrainedTokenizer, 
                          TrainerCallback, TrainerControl, TrainerState,
                          TrainingArguments)

supported_tasks = [
    "question-answering",
    "summarization",
    "table-question-answering",
    "text2text-generation",
    "text-classification",
    "sentiment-analysis",
    "text-generation",
    "token-classification",
    "translation",
    "translation_xx_to_yy",
]


class ConfigKeys:
    deepspeed = "deepspeed"
    quantization = "quantization"
    training = "training"
    tokenizer_pretrained = "tokenizer_pretrained"
    model_pretrained = "model_pretrained"
    peft_config = "peft_config"
    data_collator = "data_collator"
    beta = "beta"


# ----------------------from MLRUN--------------------------------
class HFTrainerMLRunInterface(MLRunInterface, ABC):
    """
    This is temporary and will be built in mlrun 1.5.0
    Interface for adding MLRun features for tensorflow keras API.
    """

    # MLRuns context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-huggingface"

    # Attributes to replace so the MLRun interface will be fully enabled.
    _REPLACED_METHODS = [
        "train",
        # "evaluate"
    ]

    @classmethod
    def add_interface(
        cls,
        obj: DPOTrainer,
        restoration: CommonTypes.MLRunInterfaceRestorationType = None,
    ):
        super(HFTrainerMLRunInterface, cls).add_interface(
            obj=obj, restoration=restoration
        )

    @classmethod
    def mlrun_train(cls):
        def wrapper(self: DPOTrainer, *args, **kwargs):
            # Restore the evaluation method as `train` will use it:
            # cls._restore_attribute(obj=self, attribute_name="evaluate")

            # Call the original fit method:
            result = self.original_train(*args, **kwargs)

            # Replace the evaluation method again:
            # cls._replace_function(obj=self, function_name="evaluate")

            return result

        return wrapper


class MLRunCallback(TrainerCallback):
    """
    This is temporary and will be built in mlrun 1.5.0
    Callback for collecting logs during training / evaluation of the `Trainer` API.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        model_name: str = "model",
        tag: str = "",
        labels: Dict[str, str] = None,
        extra_data: dict = None,
    ):
        super().__init__()

        # Store the configurations:
        self._context = (
            context
            if context is not None
            else mlrun.get_or_create_ctx("./mlrun-huggingface")
        )
        self._model_name = model_name
        self._tag = tag
        self._labels = labels
        self._extra_data = extra_data if extra_data is not None else {}

        # Set up the logging mode:
        self._is_training = False
        self._steps: List[List[int]] = []
        self._metric_scores: Dict[str, List[float]] = {}
        self._artifacts: Dict[str, Artifact] = {}

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._steps.append([])

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self.log_metrics()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        recent_logs = state.log_history[-1].copy()

        recent_logs.pop("epoch")
        current_step = int(recent_logs.pop("step"))
        if current_step not in self._steps[-1]:
            self._steps[-1].append(current_step)

        for metric_name, metric_score in recent_logs.items():
            if metric_name.startswith("train_"):
                if metric_name.split("train_")[1] not in self._metric_scores:
                    self._metric_scores[metric_name] = [metric_score]
                continue
            if metric_name not in self._metric_scores:
                self._metric_scores[metric_name] = []
            self._metric_scores[metric_name].append(metric_score)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._is_training = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizer = None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self.log_metrics()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self.log_metrics()

        if self._is_training:
            return

    def log_metrics(self):
        for metric_name, metric_scores in self._metric_scores.items():
            self._context.log_result(key=metric_name, value=metric_scores[-1])
            if len(metric_scores) > 1:
                self.log_metric_plot(name=metric_name, scores=metric_scores)
        self._context.commit(completed=False)

    def log_metric_plot(self, name: str, scores: List[float]):
        # Initialize a plotly figure:
        metric_figure = go.Figure()

        # Add titles:
        metric_figure.update_layout(
            title=name.capitalize().replace("_", " "),
            xaxis_title="Samples",
            yaxis_title="Scores",
        )

        # Draw:
        metric_figure.add_trace(
            go.Scatter(x=np.arange(len(scores)), y=scores, mode="lines")
        )

        # Create the plotly artifact:
        artifact_name = f"{name}_plot"
        artifact = PlotlyArtifact(key=artifact_name, figure=metric_figure)
        self._artifacts[artifact_name] = self._context.log_artifact(artifact)


def apply_mlrun(
    trainer: DPOTrainer,
    model_name: str = None,
    tag: str = "",
    context: mlrun.MLClientCtx = None,
    auto_log: bool = True,
    labels: Dict[str, str] = None,
    extra_data: dict = None,
    **kwargs,
):
    """
    This is temporary and will be built in mlrun 1.5.0
    """
    # Get parameters defaults:
    if context is None:
        context = mlrun.get_or_create_ctx(HFTrainerMLRunInterface.DEFAULT_CONTEXT_NAME)

    HFTrainerMLRunInterface.add_interface(obj=trainer)

    if auto_log:
        trainer.add_callback(
            MLRunCallback(
                context=context,
                model_name=model_name,
                tag=tag,
                labels=labels,
                extra_data=extra_data,
            )
        )


# ----------------------end from MLRUN--------------------------------


def _print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%:"
        f" {100 * trainable_params / all_param}"
    )


# default configs
# will be used if user provides "True" with config name as input
QUANTIZATION_CONFIG = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

PEFT_CONFIG = peft.LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

DEEPSPEED_CONFIG = {
    "train_micro_batch_size_per_gpu": "auto",
    "fp16": {"enabled": True},
    "autotuning": {
        "enabled": True,
        "arg_mappings": {
            "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
            "gradient_accumulation_steps ": "--gradient_accumulation_steps",
        },
    },
    "zero_optimization": {
        "stage": 2,
    },
}


def _update_config(src: dict, dst: dict):
    """
    update configs according to user, this way the user can add/modify values in default configs for e.g.

    goes over all configs and corresponding prefixes, collect all the keys from the given dict that start
     with the prefix and add them to appropriate config

    :param src: dict of all candidate values to update dict.
    :param dst: dict containing all configs to update.
    """

    for config_name, config in dst.items():

        # If given True we use default dict
        # Can also be False or a config dict given from user, so we check specifically fo True
        if config is True and config_name == "quantization":
            config = QUANTIZATION_CONFIG

        if config is True and config_name == "lora":
            config = PEFT_CONFIG

        if config is True and config_name == "deepspeed":
            config = DEEPSPEED_CONFIG

        # in some cases we can get a boolean value, in that case no need to look for args
        if isinstance(config, bool):
            config = None

        elif isinstance(config, dict):
            for key, val in src.items():
                if key.startswith(config_name):
                    config[key.replace(f"{config_name}_", "")] = val

        # update by config name
        else:
            for key, val in src.items():
                if key.startswith(config_name):
                    setattr(config, key.replace(f"{config_name}_", ""), val)

        dst.update({config_name: config})


def _get_class_object(class_path: str) -> type:
    """
    given a full class name, this function returns the correct class

    :param class_path: a full class name (ex. 'transformers.AutoModelForCausalLM')

    :return the wanted class object
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _set_model_and_tokenizer(
    model: Union[str, List[str]],
    tokenizer: Union[str, List[str]],
    task: str,
    framework: str,
    quantization_config: dict,
    use_cuda: bool,
    tokenizer_pretrained_config,
    model_pretrained_config,
    device_map: str,
):
    """
    get the correct model and tokenizer according to given user inputs

    :param model: a tuple containing model name and class, or str with model name or path
    :param tokenizer: a tuple containing tokenizer name and class, or str with tokenizer name or path
    :param task: a supported nlp task, used to choose model if not provided
    :param framework: pt or tf
    :param quantization_config: quantization config or None, to load model in appropriate way
    :param use_cuda: use gpu or not
    :param tokenizer_pretrained_config: config to load the pretrained tokenizer
    :param model_pretrained_config: config to load the pretrained model
    :param device_map: a device map for model training if using number of gpu's

    :returns: model and tokenizer
    """
    # if task is not supported and no model was given we can't choose one
    if task and task not in supported_tasks and not model:
        logger.error("unsupported task option chosen")
        raise

    # load model from store
    if isinstance(model, str) and is_store_uri(model):
        pass
        # TODO: load both model and tokenizer and return, need guy's help

    # if it's a tuple them we assume it contains of both name and class
    if isinstance(model, list):
        model_name, model_class = model
        model_class = _get_class_object(model_class)

    # in the case we don't get the model class we need the task in order to choose the correct model
    else:
        if task is None:
            logger.error("task must be chosen in order to determine the correct model")
            raise Exception(
                "this function requires either a supported task or a model and model class to be chosen"
            )

        _, available_classes, task_options = transformers.pipelines.check_task(task)

        if isinstance(model, str):
            model_name = model

        # if model is not given, we take the default model for the given task
        else:
            model_name, _ = transformers.pipelines.get_default_model_and_revision(
                available_classes, framework, task_options
            )
        if not available_classes.get(framework, tuple()):
            logger.error(
                "given task's default model is not supported in specified framework"
            )
            raise Exception(
                "this function requires either a supported task or a model and model class to be chosen"
            )

        model_class = available_classes[framework][0]

    # load the pretrained model
    if use_cuda:
        device_map = device_map
    else:
        device_map = None

    model = model_class.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        **model_pretrained_config,
    )

    # If quantization config is given we will load a quantized model, if not a regular one
    if quantization_config:
        model.gradient_checkpointing_enable()
        model = peft.prepare_model_for_kbit_training(model)

    # if not specified we choose the default tokenizer that corresponding to the model
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        return model_name, model, tokenizer

    if isinstance(tokenizer, str):
        tokenizer_name = tokenizer
        tokenizer_class = transformers.AutoTokenizer

    # if it's not a str then it's a tuple of both name and class
    else:
        tokenizer_name, tokenizer_class = tokenizer
        tokenizer_class = _get_class_object(tokenizer_class)

    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name, **tokenizer_pretrained_config
    )

    tokenizer.pad_token = tokenizer.eos_token

    return model_name, model, tokenizer


def _dataset_loader(dataset: str, is_train: bool = True, **kwargs) -> Dataset:
    """
    loads the specific dataset provided by the user

    :param dataset: name or path of dataset to load
    :param is_train: bool that indicates the purpose of the dataset
    :param kwargs: other kwargs for loading the dataset

    :returns: loaded dataset
    """
    # if split in kwargs then the user decides how to split the dataset
    if "split" in kwargs:
        return load_dataset(dataset, **kwargs)

    # if it's a dataset for train we split with train
    if is_train:
        return load_dataset(dataset, split="train", **kwargs)

    # if it's eval dataset, then a lot of names are acceptable for the set and we check all of them
    dataset = load_dataset(dataset, **kwargs)
    if "test" in dataset:
        return dataset.get("test")
    elif "eval" in dataset:
        return dataset.get("eval")
    elif "validation" in dataset:
        return dataset.get("validation")


def _prepare_dataset(
    train_dataset: str,
    eval_dataset: str,
    train_load_dataset_kwargs,
    eval_load_dataset_kwargs,
    tokenizer,
    dataset_columns_to_train: Union[str, list],
) -> (Dataset, Union[Dataset, None]):
    """
    Loads the train and eval datasets (if provided) passes them through the tokenizer and
    returns them ready to use in training

    :param train_dataset: the name or path to the train dataset
    :param eval_dataset: the name or path to the eval dataset
    :param dataset_columns_to_train: which columns to pass to the model as inputs
                                        (need to pass through the tokenizer first)
    :param train_load_dataset_kwargs: kwargs for dataset loading
    :param eval_load_dataset_kwargs: kwargs for dataset loading
    :param tokenizer: the tokenizer to pass the data through

    :returns: tokenized datasets
    """
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # we take col name/s in a list for easy generalization
    if isinstance(dataset_columns_to_train, str):
        dataset_columns_to_train = [dataset_columns_to_train]

    if isinstance(train_dataset, mlrun.datastore.DataItem):
        train_dataset = Dataset.from_pandas(train_dataset.as_df())
        return (
            train_dataset.map(
                lambda examples: tokenizer(
                    *[examples[col] for col in dataset_columns_to_train],
                    truncation=True,
                    padding=True,
                ),
                batched=True,
            ),
            None,
        )

    # Load datasets
    # if provided two paths/names we load each separately using designated func
    if eval_dataset:
        train_dataset = _dataset_loader(
            dataset=train_dataset, is_train=True, **train_load_dataset_kwargs
        )
        eval_dataset = _dataset_loader(
            dataset=eval_dataset, is_train=False, **eval_load_dataset_kwargs
        )

    # if only on path is given then we must check if it contains both dataset or if only one should be used
    else:
        dataset = load_dataset(train_dataset, **train_load_dataset_kwargs)
        if "train" in dataset:
            train_dataset = dataset.get("train")
            if "test" in dataset:
                eval_dataset = dataset.get("test")
            elif "eval" in dataset:
                eval_dataset = dataset.get("eval")
            elif "validation" in dataset:
                eval_dataset = dataset.get("validation")
            else:
                # only train dataset given, tokenize and return it
                return (
                    train_dataset.map(
                        lambda examples: tokenizer(
                            *[examples[col] for col in dataset_columns_to_train],
                            truncation=True,
                            padding=True,
                        ),
                        batched=True,
                    ),
                    None,
                )
        else:
            logger.error("train dataset is mandatory")
            raise KeyError("no train dataset found in given dataset")

    # Tokenize the data so the model can understand it
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenizer(
            *[examples[col] for col in dataset_columns_to_train],
            truncation=True,
            padding=True,
        ),
        batched=True,
    )

    tokenized_eval_dataset = eval_dataset.map(
        lambda examples: tokenizer(
            *[examples[col] for col in dataset_columns_to_train],
            truncation=True,
            padding=True,
        ),
        batched=True,
    )

    return tokenized_train_dataset, tokenized_eval_dataset


def dpo_train(
    context: mlrun.MLClientCtx,
    train_dataset: Union[str, mlrun.datastore.DataItem],
    eval_dataset: str = None,
    train_load_dataset_kwargs: dict = {},
    eval_load_dataset_kwargs: dict = {},
    dataset_columns_to_train: Union[str, list] = "text",
    model: Union[str, List[str]] = "huggingface-model",
    tokenizer: Union[str, List[str]] = None,
    deepspeed_config: Union[dict, bool] = False,
    quantization_config: Union[dict, bool] = False,
    peft_config: Union[dict, bool] = False,
    beta: Union[float, bool] = False,
    training_config: dict = {},
    model_pretrained_config: dict = {},
    tokenizer_pretrained_config: dict = {},
    data_collator_config: dict = {},
    task: str = "text-generation",
    use_cuda: bool = True,
    framework: str = "pt",
    device_map: str = "auto",
    **kwargs,
):
    """
    Fine-tunes a Language Model (LLM) on a specific task using the provided dataset.
     The function takes various configuration parameters to customize the training process
     and adapt the model to specific tasks using a provided dataset.

    :param context: mlrun context in order to log trained model
    :param dataset_columns_to_train: which columns to pass to the model as inputs
    :param eval_load_dataset_kwargs: kwargs for dataset loading
    :param train_load_dataset_kwargs: kwargs for dataset loading
    :param framework: pt ot tf
    :param use_cuda: use gpu or not
    :param tokenizer_pretrained_config: config to load the pretrained tokenizer
    :param model_pretrained_config: config to load the pretrained model
    :param tokenizer: a tuple containing tokenizer name and class, or str with tokenizer name or path
    :param model: a tuple containing model name and class, or str with model name or path
    :param train_dataset: The train dataset used for fine-tuning the language model.
    :param eval_dataset: The eval dataset used for evaluate the language model during training.
    :param deepspeed_config: Configuration options for DeepSpeed (optional).
    :param quantization_config: Configuration options for model quantization (optional).
    :param lora_config: Configuration options for Low-Rank Approximation (LoRA) (optional).
    :param training_config: Configuration options specific to the fine-tuning training process (optional).
    :param data_collator_config: Configuration options for data collation during training (optional).
    :param task: A description of the specific task the model is being fine-tuned for.
    :param kwargs: Additional keyword arguments.
    """

    # TODO: match forward.keyword to dataset.keyword - check if relevant in new design
    # TODO: add warning for label, and add option to modify dataset col names - check if relevant in new design
    # Look for updates to configs given in kwargs
    configs = {
        ConfigKeys.deepspeed: deepspeed_config,
        ConfigKeys.quantization: quantization_config,
        ConfigKeys.training: training_config,
        ConfigKeys.model_pretrained: model_pretrained_config,
        ConfigKeys.tokenizer_pretrained: tokenizer_pretrained_config,
        ConfigKeys.data_collator: data_collator_config,
        ConfigKeys.peft_config: peft_config,
        ConfigKeys.beta: beta,
    }
    _update_config(dst=configs, src=kwargs)

    # check gpu permission and availability
    if use_cuda:
        if torch.cuda.is_available():
            # Clean gpu cache
            torch.cuda.empty_cache()
        else:
            logger.warning("'use_cuda' is set to True, but no cuda device is available")

    # get model and tokenizer
    model_name, model, tokenizer = _set_model_and_tokenizer(
        model=model,
        tokenizer=tokenizer,
        task=task,
        framework=framework,
        quantization_config=configs[ConfigKeys.quantization],
        use_cuda=use_cuda,
        tokenizer_pretrained_config=tokenizer_pretrained_config,
        model_pretrained_config=configs[ConfigKeys.model_pretrained],
        device_map=device_map,
    )
    whole_dataset = load_dataset(train_dataset, split='train')
    whole_dataset = whole_dataset.shuffle(seed=42).train_test_split(seed=42, test_size=.3)
    train_dataset =  whole_dataset['train']
    eval_dataset = whole_dataset['test']
    # Load datasets
    #tokenized_train, tokenized_eval = _prepare_dataset(
    #    train_dataset=train_dataset,
    #    eval_dataset=eval_dataset,
    #    train_load_dataset_kwargs=train_load_dataset_kwargs,
    #    eval_load_dataset_kwargs=eval_load_dataset_kwargs,
    #    tokenizer=tokenizer,
    #    dataset_columns_to_train=dataset_columns_to_train,
    #)

    # Initialize the data collator for the trainer to use in order to create batches of data
    #data_collator = transformers.DataCollatorForLanguageModeling(
    #    tokenizer=tokenizer, mlm=False, **data_collator_config
    #)

    # Initialize training kwargs from user kwargs:
    train_kwargs = configs[ConfigKeys.training]

    # If deepspeed config given we add it to training kwargs
    if configs[ConfigKeys.deepspeed]:
        train_kwargs["deepspeed"] = configs[ConfigKeys.deepspeed]

    # Take a look at the trainable parameters in the model
    _print_trainable_parameters(model)

    # Preparing training arguments:
    training_args = transformers.TrainingArguments(
        output_dir=tempfile.mkdtemp(),
        **train_kwargs,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model = None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=configs[ConfigKeys.peft_config],
        beta = configs[ConfigKeys.beta],
        tokenizer=tokenizer,
        #data_collator=data_collator,
        args=training_args,
    )

    apply_mlrun(trainer, model_name=model_name.split("/")[-1])
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    # Apply training with evaluation:
    context.logger.info(f"training '{model_name}'")
    trainer.train()

    temp_directory = tempfile.TemporaryDirectory().name
    trainer.save_model(temp_directory)

    # Zip the model directory:
    shutil.make_archive(
        base_name="model",
        format="zip",
        root_dir=temp_directory,
    )

    # Log the model:
    context.log_model(
        key="model",
        db_key=model_name.split("/")[-1],
        model_file="model.zip",
        tag="",
        framework="Hugging Face",
    )


def evaluate(
    context,
    model_path,
    data: pd.DataFrame,
    model_name: str = None,
    tokenizer_name: str = None,
):
    """
    Evaluating the model using perplexity, for more information visit:
    https://huggingface.co/docs/transformers/perplexity

    :param context:     mlrun context
    :param model_path:  path to the model directory
    :param data:        the data to evaluate the model
    :param model_name:  name of base model
    :param tokenizer_name: name of base tokenizer
    """
    # Get the model artifact and file:
    (
        model_file,
        model_artifact,
        extra_data,
    ) = mlrun.artifacts.get_model(model_path)

    # Read the name:
    _model_name = model_artifact.spec.db_key

    # Extract logged model files:
    model_directory = os.path.join(os.path.dirname(model_file), _model_name)
    with zipfile.ZipFile(model_file, "r") as zip_file:
        zip_file.extractall(model_directory)

    # Loading the saved pretrained tokenizer and model:
    dataset = Dataset.from_pandas(data)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda:0", trust_remote_code=True, load_in_8bit=True
    )
    model = PeftModel.from_pretrained(model, model_directory)
    model.eval()
    encodings = tokenizer("\n\n".join(dataset["text"][:5]), return_tensors="pt")

    max_length = 1024
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids.cuda(), labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    context.log_result("perplexity", ppl)
