import os
import shutil
import tempfile
from abc import ABC
from typing import Any, Dict, List, Optional

import mlrun
import numpy as np
import transformers
from datasets import Dataset, load_dataset, load_metric
from mlrun import MLClientCtx
from mlrun.artifacts import Artifact, PlotlyArtifact
from mlrun.frameworks._common import CommonTypes, MLRunInterface
from mlrun.utils import create_class
from plotly import graph_objects as go
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          PreTrainedModel, PreTrainedTokenizer, Trainer,
                          TrainerCallback, TrainerControl, TrainerState,
                          TrainingArguments)


# ----------------------from MLRUN--------------------------------
class HFTrainerMLRunInterface(MLRunInterface, ABC):
    """
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
        obj: Trainer,
        restoration: CommonTypes.MLRunInterfaceRestorationType = None,
    ):
        """
        Enrich the object with this interface properties, methods and functions, so it will have this TensorFlow.Keras
        MLRuns features.
        :param obj:                     The object to enrich his interface.
        :param restoration: Restoration information tuple as returned from 'remove_interface' in order to
                                        add the interface in a certain state.
        """

        super(HFTrainerMLRunInterface, cls).add_interface(
            obj=obj, restoration=restoration
        )

    @classmethod
    def mlrun_train(cls):

        """
        MLRuns tf.keras.Model.fit wrapper. It will setup the optimizer when using horovod. The optimizer must be
        passed in a keyword argument and when using horovod, it must be passed as an Optimizer instance, not a string.

        raise MLRunInvalidArgumentError: In case the optimizer provided did not follow the instructions above.
        """

        def wrapper(self: Trainer, *args, **kwargs):
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
        self._steps.append([])

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._log_metrics()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs,
    ):
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
        self._log_metrics()

        temp_directory = tempfile.gettempdir()

        # Save and log the tokenizer:
        if tokenizer is not None:
            # Save tokenizer:
            tokenizer_dir = os.path.join(temp_directory, "tokenizer")
            tokenizer.save_pretrained(save_directory=tokenizer_dir)
            # Zip the tokenizer directory:
            tokenizer_zip = shutil.make_archive(
                base_name="tokenizer",
                format="zip",
                root_dir=tokenizer_dir,
            )
            # Log the zip file:
            self._artifacts["tokenizer"] = self._context.log_artifact(
                item="tokenizer", local_path=tokenizer_zip
            )

        # Save the model:
        model_dir = os.path.join(temp_directory, "model")
        model.save_pretrained(save_directory=model_dir)

        # Zip the model directory:
        shutil.make_archive(
            base_name="model",
            format="zip",
            root_dir=model_dir,
        )

        # Log the model:
        self._context.log_model(
            key="model",
            db_key=self._model_name,
            model_file="model.zip",
            tag=self._tag,
            framework="Hugging Face",
            labels=self._labels,
            extra_data={**self._artifacts, **self._extra_data},
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._log_metrics()

        if self._is_training:
            return

        # TODO: Update the model object

    def _log_metrics(self):
        for metric_name, metric_scores in self._metric_scores.items():
            self._context.log_result(key=metric_name, value=metric_scores[-1])
            if len(metric_scores) > 1:
                self._log_metric_plot(name=metric_name, scores=metric_scores)
        self._context.commit(completed=False)

    def _log_metric_plot(self, name: str, scores: List[float]):
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
    huggingface_object: transformers.Trainer,
    model_name: str = None,
    tag: str = "",
    context: mlrun.MLClientCtx = None,
    auto_log: bool = True,
    labels: Dict[str, str] = None,
    extra_data: dict = None,
    **kwargs,
):
    """
    Wrap the given model with MLRun's interface providing it with mlrun's additional features.
    :param huggingface_object: The model to wrap. Can be loaded from the model path given as well.
    :param model_name:         The model name to use for storing the model artifact. Default: "model".
    :param tag:                The model's tag to log with.
    :param context:            MLRun context to work with. If no context is given it will be retrieved via
                               'mlrun.get_or_create_ctx(None)'
    :param auto_log:           Whether to enable MLRun's auto logging. Default: True.
    """
    if isinstance(huggingface_object, transformers.Trainer):
        if context is None:
            context = mlrun.get_or_create_ctx(
                HFTrainerMLRunInterface.DEFAULT_CONTEXT_NAME
            )

        HFTrainerMLRunInterface.add_interface(obj=huggingface_object)

        if auto_log:
            huggingface_object.add_callback(
                MLRunCallback(
                    context=context,
                    model_name=model_name,
                    tag=tag,
                    labels=labels,
                    extra_data=extra_data,
                )
            )
        return
    raise mlrun.errors.MLRunInvalidArgumentError


# ---------------------- from auto_trainer--------------------------------
class KWArgsPrefixes:
    MODEL_CLASS = "CLASS_"
    FIT = "FIT_"
    TRAIN = "TRAIN_"
    PREDICT = "PREDICT_"


def _get_sub_dict_by_prefix(src: Dict, prefix_key: str) -> Dict[str, Any]:
    """
    Collect all the keys from the given dict that starts with the given prefix and creates a new dictionary with these
    keys.

    :param src:         The source dict to extract the values from.
    :param prefix_key:  Only keys with this prefix will be returned. The keys in the result dict will be without this
                        prefix.
    """
    return {
        key.replace(prefix_key, ""): val
        for key, val in src.items()
        if key.startswith(prefix_key)
    }


# ---------------------- Hugging Face Trainer --------------------------------


def _create_compute_metrics(metrics: List[str]):
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metric_dict_results = {}
        for metric in metrics:
            load_met = load_metric(metric)
            metric_res = load_met.compute(predictions=predictions, references=labels)[
                metric
            ]
            metric_dict_results[metric] = metric_res

        return metric_dict_results

    return _compute_metrics


def _edit_columns(
    dataset: Dataset,
    drop_columns: List[str] = None,
    rename_columns: [str, str] = None,
):
    if drop_columns:
        dataset = dataset.remove_columns(drop_columns)
    if rename_columns:
        dataset = dataset.rename_columns(rename_columns)
    return dataset


def _prepare_dataset(
    context: MLClientCtx,
    dataset_name: str,
    label_name: str = None,
    drop_columns: Optional[List[str]] = None,
    num_of_train_samples: int = None,
    train_test_split_size: float = None,
    random_state: int = None,
):
    """
    Loading the dataset and editing the columns

    :param context:                 MLRun contex
    :param dataset_name:            The name of the dataset to get from the HuggingFace hub
    :param label_name:
    :param drop_columns:            The columns to drop from the dataset.
    :param num_of_train_samples:    Max number of training samples, for debugging.
    :param train_test_split_size:   Should be between 0.0 and 1.0 and represent the proportion of the dataset to include
                                    in the test split.
    :param random_state:            Random state for train_test_split

    """

    context.logger.info(
        f"Loading and editing {dataset_name} dataset from Hugging Face hub"
    )
    rename_cols = {label_name: "labels"}

    # Loading and editing dataset:
    dataset = load_dataset(dataset_name)

    # train set
    train_dataset = dataset["train"]
    if num_of_train_samples:
        train_dataset = train_dataset.shuffle(seed=random_state).select(
            list(range(num_of_train_samples))
        )
    train_dataset = _edit_columns(train_dataset, drop_columns, rename_cols)

    # test set
    test_dataset = dataset["test"]
    if train_test_split_size or num_of_train_samples:
        train_test_split_size = train_test_split_size or 0.2
        num_of_test_samples = int(
            (train_dataset.num_rows * train_test_split_size)
            // (1 - train_test_split_size)
        )
        test_dataset = test_dataset.shuffle(seed=random_state).select(
            list(range(num_of_test_samples))
        )
    test_dataset = _edit_columns(test_dataset, drop_columns, rename_cols)

    return train_dataset, test_dataset


def train(
    context: MLClientCtx,
    dataset_name: str = None,
    drop_columns: Optional[List[str]] = None,
    pretrained_tokenizer: str = None,
    pretrained_model: str = None,
    model_class: str = None,
    model_name: str = "huggingface_model",
    label_name: str = "labels",
    text_col: str = "text",
    num_of_train_samples: int = None,
    train_test_split_size: float = None,
    metrics: List[str] = None,
    random_state: int = None,
):
    """
    Training and evaluating a pretrained model with a pretrained tokenizer over a dataset.

    :param context:                 MLRun context
    :param dataset_name:            The name of the dataset to get from the HuggingFace hub
    :param drop_columns:            The columns to drop from the dataset.
    :param pretrained_tokenizer:    The name of the pretrained tokenizer from the HuggingFace hub.
    :param pretrained_model:        The name of the pretrained model from the HuggingFace hub.
    :param model_name:              The model's name to use for storing the model artifact, default to 'model'
    :param model_class:             The class of the model, e.g. `transformers.AutoModelForSequenceClassification`
    :param label_name:              The target label of the column in the dataset.
    :param text_col:                The input text column un the dataset.
    :param num_of_train_samples:    Max number of training samples, for debugging.
    :param train_test_split_size:   Should be between 0.0 and 1.0 and represent the proportion of the dataset to include
                                    in the test split.
    :param metrics:                 List of different metrics for evaluate the model such as f1, accuracy etc.
    :param random_state:            Random state for train_test_split
    """

    if not label_name or not pretrained_tokenizer:
        raise mlrun.errors.MLRunRuntimeError(
            "Must provide label_names and pretrained_tokenizer"
        )

    # Creating tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    def preprocess_function(examples):
        return tokenizer(examples[text_col], truncation=True)

    # prepare data for training
    train_dataset, test_dataset = _prepare_dataset(
        context,
        dataset_name,
        label_name,
        drop_columns,
        num_of_train_samples,
        train_test_split_size,
        random_state=random_state,
    )

    # Mapping datasets with the tokenizer:
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    # Creating data collator for batching:
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Parsing kwargs:
    train_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.TRAIN
    )
    model_class_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.MODEL_CLASS
    )

    # Loading our pretrained model:
    model_class_kwargs["pretrained_model_name_or_path"] = (
        model_class_kwargs.get("pretrained_model_name_or_path") or pretrained_model
    )
    if not model_class_kwargs["pretrained_model_name_or_path"]:
        raise mlrun.errors.MLRunRuntimeError(
            "Must provide pretrained_model name as "
            "function argument or in extra params"
        )
    model = create_class(model_class).from_pretrained(**model_class_kwargs)

    # Preparing training arguments:
    training_args = TrainingArguments(
        **train_kwargs,
    )

    compute_metrics = _create_compute_metrics(metrics) if metrics else None
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    apply_mlrun(trainer, model_name=model_name)

    # Apply training with evaluation:
    context.logger.info(f"training '{model_name}'")
    trainer.train()
