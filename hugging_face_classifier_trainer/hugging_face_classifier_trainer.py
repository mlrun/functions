import os
import shutil
import tempfile
import zipfile
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Union

import mlrun
import numpy as np
import pandas as pd
import transformers
from datasets import Dataset, load_dataset, load_metric
from mlrun import MLClientCtx
from mlrun import feature_store as fs
from mlrun.api.schemas import ObjectKind
from mlrun.artifacts import Artifact, PlotlyArtifact
from mlrun.datastore import DataItem
from mlrun.frameworks._common import CommonTypes, MLRunInterface
from mlrun.utils import create_class
from plotly import graph_objects as go
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


# ----------------------from MLRUN--------------------------------
class HFORTOptimizerMLRunInterface(MLRunInterface, ABC):
    """
    Interface for adding MLRun features for tensorflow keras API.
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-huggingface"

    # Attributes to be inserted so the MLRun interface will be fully enabled.
    _PROPERTIES = {
        "_auto_log": False,
        "_context": None,
        "_model_name": "model",
        "_tag": "",
        "_labels": None,
        "_extra_data": None,
    }
    _METHODS = ["enable_auto_logging"]
    # Attributes to replace so the MLRun interface will be fully enabled.
    _REPLACED_METHODS = [
        "optimize",
    ]

    @classmethod
    def add_interface(
        cls,
        obj,
        restoration: CommonTypes.MLRunInterfaceRestorationType = None,
    ):
        """
        Enrich the object with this interface properties, methods and functions, so it will have this TensorFlow.Keras
        MLRun's features.
        :param obj:                     The object to enrich his interface.
        :param restoration: Restoration information tuple as returned from 'remove_interface' in order to
                                        add the interface in a certain state.
        """
        super(HFORTOptimizerMLRunInterface, cls).add_interface(
            obj=obj, restoration=restoration
        )

    @classmethod
    def mlrun_optimize(cls):
        """
        MLRun's tf.keras.Model.fit wrapper. It will setup the optimizer when using horovod. The optimizer must be
        passed in a keyword argument and when using horovod, it must be passed as an Optimizer instance, not a string.

        raise MLRunInvalidArgumentError: In case the optimizer provided did not follow the instructions above.
        """

        def wrapper(self, *args, **kwargs):
            save_dir = cls._get_function_argument(
                self.optimize,
                argument_name="save_dir",
                passed_args=args,
                passed_kwargs=kwargs,
            )[0]

            # Call the original optimize method:
            result = self.original_optimize(*args, **kwargs)

            if self._auto_log:
                # Zip the onnx model directory:
                shutil.make_archive(
                    base_name="onnx_model",
                    format="zip",
                    root_dir=save_dir,
                )
                # Log the onnx model:
                self._context.log_model(
                    key="model",
                    db_key=self._model_name,
                    model_file="onnx_model.zip",
                    tag=self._tag,
                    framework="ONNX",
                    labels=self._labels,
                    extra_data=self._extra_data,
                )

            return result

        return wrapper

    def enable_auto_logging(
        self,
        context: mlrun.MLClientCtx,
        model_name: str = "model",
        tag: str = "",
        labels: Dict[str, str] = None,
        extra_data: dict = None,
    ):
        self._auto_log = True

        self._context = context
        self._model_name = model_name
        self._tag = tag
        self._labels = labels
        self._extra_data = extra_data


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


def _apply_mlrun_on_trainer(
    trainer: transformers.Trainer,
    model_name: str = None,
    tag: str = "",
    context: mlrun.MLClientCtx = None,
    auto_log: bool = True,
    labels: Dict[str, str] = None,
    extra_data: dict = None,
    **kwargs,
):
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


def _apply_mlrun_on_optimizer(
    optimizer,
    model_name: str = None,
    tag: str = "",
    context: mlrun.MLClientCtx = None,
    auto_log: bool = True,
    labels: Dict[str, str] = None,
    extra_data: dict = None,
    **kwargs,
):
    # Get parameters defaults:
    if context is None:
        context = mlrun.get_or_create_ctx(
            HFORTOptimizerMLRunInterface.DEFAULT_CONTEXT_NAME
        )

    HFORTOptimizerMLRunInterface.add_interface(obj=optimizer)

    if auto_log:
        optimizer.enable_auto_logging(
            context=context,
            model_name=model_name,
            tag=tag,
            labels=labels,
            extra_data=extra_data,
        )


def apply_mlrun(
    huggingface_object,
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
        return _apply_mlrun_on_trainer(
            trainer=huggingface_object,
            model_name=model_name,
            tag=tag,
            context=context,
            auto_log=auto_log,
            labels=labels,
            extra_data=extra_data,
        )
    import optimum.onnxruntime as optimum_ort

    if isinstance(huggingface_object, optimum_ort.ORTOptimizer):
        return _apply_mlrun_on_optimizer(
            optimizer=huggingface_object,
            model_name=model_name,
            tag=tag,
            context=context,
            auto_log=auto_log,
            labels=labels,
            extra_data=extra_data,
        )
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


def _get_dataframe(
    context: MLClientCtx,
    dataset: DataItem,
    label_columns: Optional[Union[str, List[str]]] = None,
    drop_columns: Union[str, List[str], int, List[int]] = None,
) -> Tuple[pd.DataFrame, Optional[Union[str, List[str]]]]:
    """
    Getting the DataFrame of the dataset and drop the columns accordingly.

    :param context:         MLRun context.
    :param dataset:         The dataset to train the model on.
                            Can be either a list of lists, dict, URI or a FeatureVector.
    :param label_columns:   The target label(s) of the column(s) in the dataset. for Regression or
                            Classification tasks.
    :param drop_columns:    str/int or a list of strings/ints that represent the column names/indices to drop.
    """
    if isinstance(dataset, (list, dict)):
        dataset = pd.DataFrame(dataset)
        # Checking if drop_columns provided by integer type:
        if drop_columns:
            if isinstance(drop_columns, str) or (
                isinstance(drop_columns, list)
                and any(isinstance(col, str) for col in drop_columns)
            ):
                context.logger.error(
                    "drop_columns must be an integer/list of integers if not provided with a URI/FeatureVector dataset"
                )
                raise ValueError
            dataset.drop(drop_columns, axis=1, inplace=True)

        return dataset, label_columns

    if dataset.meta and dataset.meta.kind == ObjectKind.feature_vector:
        # feature-vector case:
        label_columns = label_columns or dataset.meta.status.label_column
        dataset = fs.get_offline_features(
            dataset.meta.uri, drop_columns=drop_columns
        ).to_dataframe()

        context.logger.info(f"label columns: {label_columns}")
    else:
        # simple URL case:
        dataset = dataset.as_df()
        if drop_columns:
            if all(col in dataset for col in drop_columns):
                dataset = dataset.drop(drop_columns, axis=1)
            else:
                context.logger.info(
                    "not all of the columns to drop in the dataset, drop columns process skipped"
                )
    return dataset, label_columns


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
    hf_dataset: str = None,
    dataset: DataItem = None,
    drop_columns: Optional[List[str]] = None,
    pretrained_tokenizer: str = None,
    pretrained_model: str = None,
    model_class: str = None,
    model_name: str = "huggingface-model",
    label_name: str = "labels",
    text_col: str = "text",
    num_of_train_samples: int = None,
    train_test_split_size: float = None,
    metrics: List[str] = None,
    random_state: int = None,
):
    """
    Training and evaluating a pretrained model with a pretrained tokenizer over a dataset.
    The dataset can be either be the name of the dataset that contains in the HuggingFace hub,
    or a URI or a FeatureVector

    :param context:                 MLRun context
    :param hf_dataset:              The name of the dataset to get from the HuggingFace hub
    :param dataset:                 The dataset to train the model on. Can be either a URI or a FeatureVector
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

    if train_test_split_size is None:
        context.logger.info(
            "test_set or train_test_split_size are not provided, setting train_test_split_size to 0.2"
        )
        train_test_split_size = 0.2

    # Creating tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    def preprocess_function(examples):
        return tokenizer(examples[text_col], truncation=True)

    # prepare data for training
    if hf_dataset:
        train_dataset, test_dataset = _prepare_dataset(
            context,
            hf_dataset,
            label_name,
            drop_columns,
            num_of_train_samples,
            train_test_split_size,
            random_state=random_state,
        )
    elif dataset:
        # Get DataFrame by URL or by FeatureVector:
        dataset, label_name = _get_dataframe(
            context=context,
            dataset=dataset,
            label_columns=label_name,
            drop_columns=drop_columns,
        )

        train_dataset, test_dataset = train_test_split(
            dataset, test_size=train_test_split_size, random_state=random_state
        )
        train_dataset = Dataset.from_pandas(train_dataset)
        test_dataset = Dataset.from_pandas(test_dataset)

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


def _get_model_dir(model_uri: str):
    model_file, _, _ = mlrun.artifacts.get_model(model_uri)
    model_dir = tempfile.gettempdir()
    # Unzip the Model:
    with zipfile.ZipFile(model_file, "r") as zip_file:
        zip_file.extractall(model_dir)

    return model_dir


def optimize(
    model_path: str,
    model_name: str = "optimized_model",
    target_dir: str = "./optimized",
    optimization_level: int = 1,
):
    """
    Optimizing the transformer model using ONNX optimization.


    :param model_path:          The path of the model to optimize.
    :param model_name:          Name of the optimized model.
    :param target_dir:          The directory to save the ONNX model.
    :param optimization_level:  Optimization level performed by ONNX Runtime of the loaded graph. (default is 1)
    """
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
    from optimum.onnxruntime.configuration import OptimizationConfig

    model_dir = _get_model_dir(model_uri=model_path)
    # Creating configuration for optimization step:
    optimization_config = OptimizationConfig(optimization_level=optimization_level)

    # Converting our pretrained model to an ONNX-Runtime model:
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir, from_transformers=True
    )

    # Creating an ONNX-Runtime optimizer from ONNX model:
    optimizer = ORTOptimizer.from_pretrained(ort_model)

    apply_mlrun(optimizer, model_name=model_name)
    # Optimizing and saving the ONNX model:
    optimizer.optimize(save_dir=target_dir, optimization_config=optimization_config)
