# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import json
import logging
from typing import Tuple, List

from mlrun import MLClientCtx, DataItem, get_dataitem
import mlrun.feature_store as f_store
import mlrun.datastore
import mlrun.utils
from mlrun.datastore.targets import ParquetTarget

from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.core.dataset import Dataset
from azureml.core.model import Model
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.script_run import ScriptRun

from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun


def _env_or_secret(context, key):
    if key in os.environ:
        return os.environ[key]
    return context.get_secret(key)


def _load_workspace(context: MLClientCtx) -> Workspace:
    """
    Loading AzureML Workspace with Azure secrets.

    :param context: MLRun context.
    :returns:       AzureML Workspace
    """

    if hasattr(context, "_azure_workspace"):
        return context._azure_workspace

    context.logger.info("Loading AzureML Workspace")
    # Azure service authentication:
    service_authentication = ServicePrincipalAuthentication(
        tenant_id=_env_or_secret(context, "AZURE_TENANT_ID"),
        service_principal_id=_env_or_secret(context, "AZURE_SERVICE_PRINCIPAL_ID"),
        service_principal_password=_env_or_secret(
            context, "AZURE_SERVICE_PRINCIPAL_PASSWORD"
        ),
    )

    # Loading Azure workspace:
    workspace = Workspace(
        subscription_id=_env_or_secret(context, "AZURE_SUBSCRIPTION_ID"),
        resource_group=_env_or_secret(context, "AZURE_RESOURCE_GROUP"),
        workspace_name=_env_or_secret(context, "AZURE_WORKSPACE_NAME"),
        auth=service_authentication,
    )

    context._azure_workspace = workspace
    return workspace


def _init_experiment(
    context: MLClientCtx, experiment_name: str
) -> Tuple[Workspace, Experiment]:
    """
    Initialize workspace and experiment in Azure ML. Uses Service
    Principal authentication via environment variables.

    :param context:         MLRun context.
    :param experiment_name: Name of experiment to create in Azure ML.
    :returns:               Azure ML Workspace and Experiment.
    """

    # Initialize experiment via Service Principal Authentication:
    # https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication#use-service-principal-authentication

    workspace = _load_workspace(context)

    context.logger.info(f"Initializing AzureML experiment {experiment_name}")
    # Creating experiment:
    experiment = Experiment(workspace, experiment_name)

    return workspace, experiment


def init_compute(
    context: MLClientCtx,
    cpu_cluster_name: str,
    vm_size: str = "STANDARD_D2_V2",
    max_nodes: int = 1,
) -> ComputeTarget:
    """
    Initialize Azure ML compute target to run experiment. Checks for
    existing compute target and creates new if does not exist.

    :param context:          MLRun context.
    :param cpu_cluster_name: Name of Azure ML compute target. Created if does not exist.
    :param vm_size:          Azure machine type for compute target.
    :param max_nodes:        Maximum number of concurrent compute targets.
    :returns:                Azure ML Compute Target.
    """

    workspace = _load_workspace(context)
    context.logger.info(f"Initializing AzureML compute target {cpu_cluster_name}")

    # Verify that cluster does not exist already:
    try:
        compute_target = ComputeTarget(workspace=workspace, name=cpu_cluster_name)
        context.logger.info("Found existing cluster, will use it.")
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size, max_nodes=max_nodes
        )
        compute_target = ComputeTarget.create(
            workspace, cpu_cluster_name, compute_config
        )

    compute_target.wait_for_completion(show_output=True)
    return compute_target


def register_dataset(
    context: MLClientCtx,
    dataset_name: str,
    dataset_description: str,
    data: DataItem,
    create_new_version: bool = False,
):
    """
    Register dataset object (can be also an Iguazio FeatureVector) in Azure ML.
    Uploads parquet file to Azure blob storage and registers
    that file as a dataset in Azure ML.

    :param context:               MLRun context.
    :param dataset_name:          Name of Azure dataset to register.
    :param dataset_description:   Description of Azure dataset to register.
    :param data:                  MLRun FeatureVector or dataset object to upload.
    :param create_new_version:    Register Azure dataset as new version. Must be used when
                                  modifying dataset schema.
    """

    # test for Azure storage connection environment variable or secret:
    assert _env_or_secret(
        context, "AZURE_STORAGE_CONNECTION_STRING"
    ), "AZURE_STORAGE_CONNECTION_STRING secret not set"

    # Connect to AzureML experiment and datastore:
    context.logger.info("Connecting to AzureML experiment default datastore")

    workspace = _load_workspace(context)
    datastore = workspace.get_default_datastore()

    # Azure blob path (default datastore for workspace):
    blob_path = f"az://{datastore.container_name}/{dataset_name}"

    store_uri_prefix, _ = mlrun.datastore.parse_store_uri(data.artifact_url)
    feature_vector_case = mlrun.utils.StorePrefix.FeatureVector == store_uri_prefix
    # Retrieve data source as dataframe:
    if feature_vector_case:
        # FeatureVector case:
        context.logger.info(
            f"Retrieving feature vector and uploading to Azure blob storage: {blob_path}"
        )
        f_store.get_offline_features(data.meta.uri, target=ParquetTarget(path=blob_path))
    else:
        blob_path += data.suffix
        # DataItem case:
        context.logger.info(
            f"Retrieving feature vector and uploading to Azure blob storage: {blob_path}"
        )
        data_in_bytes = data.get()
        get_dataitem(blob_path).put(data_in_bytes)

    # Register dataset in AzureML:
    context.logger.info(f"Registering dataset {dataset_name} in Azure ML")
    if data.suffix == ".parquet" or feature_vector_case:
        dataset = Dataset.Tabular.from_parquet_files(
            path=(datastore, f"{dataset_name}.parquet"), validate=False
        )
    else:
        context.logger.info(
            f"OpenSSL version must be 1.1. Overriding the OpenSSL version to 1.1"
        )
        # OpenSSL version must be 1.1
        os.environ["CLR_OPENSSL_VERSION_OVERRIDE"] = "1.1"
        dataset = Dataset.Tabular.from_delimited_files(
            path=(datastore, f"{dataset_name}{data.suffix}"), validate=False
        )

    dataset.register(
        workspace=workspace,
        name=dataset_name,
        description=dataset_description,
        create_new_version=create_new_version,
    )

    # Output registered dataset name in Azure:
    context.log_result("dataset_blob_path", blob_path)


def download_model(
    context: MLClientCtx,
    model_name: str,
    model_version: int,
    target_dir: str = ".",
) -> None:
    """
    Download trained model from Azure ML to local filesystem.

    :param context:       MLRun context.
    :param model_name:    Name of trained and registered model.
    :param model_version: Version of model to download.
    :param target_dir:    Target directory to download model.
    """
    # Loading workspace if not provided:
    workspace = _load_workspace(context)
    context.logger.info(f"Downloading model {model_name}:{model_version}")
    model = Model(workspace, model_name, version=model_version)
    model.download(target_dir=target_dir, exist_ok=True)


def upload_model(
    context: MLClientCtx,
    model_name: str,
    model_path: str,
    model_description: str = None,
    model_tags: dict = None,
) -> None:
    """
    Upload pre-trained model from local filesystem to Azure ML.
    :param context:           MLRun context.
    :param model_name:        Name of trained and registered model.
    :param model_path:        Path to file on local filesystem.
    :param model_description: Description of models.
    :param model_tags:        KV pairs of model tags.
    """
    # Loading workspace if not provided:
    workspace = _load_workspace(context)

    context.logger.info(f"Upload model {model_name} from {model_path}")
    Model.register(
        workspace=workspace,
        model_path=model_path,
        model_name=model_name,
        description=model_description,
        tags=model_tags,
    )


def _get_top_n_runs(
    remote_run: AutoMLRun, n: int = 5, primary_metric: str = "accuracy"
) -> List[ScriptRun]:
    """
    Get top N complete runs from experiment sorted by primary metric.

    :param remote_run:     Azure ML Run.
    :param n:              Number of top runs to return.
    :param primary_metric: Metric to sort by.

    :returns:              List of top N runs sorted by primary metric.
    """
    # Collect all models:
    complete_runs = [
        run
        for run in remote_run.get_children(status="Completed")
        if not any(s in run.id for s in ["setup", "worker"])
    ]

    # Checking that the required number of runs are done:
    if len(complete_runs) < n:
        raise ValueError(f"Expected {n} runs but only received {len(complete_runs)}")

    # Sorting by the primary metric:
    sorted_runs = sorted(
        complete_runs, key=lambda run: run.get_metrics()[primary_metric], reverse=True
    )
    return sorted_runs[:n]


def _get_model_hp(
    run: ScriptRun,
) -> dict:
    """
    Get hyper-parameters of trained AzureML model.
    Combine the hyper-parameters of the data transformation and training to a dictionary.
    The prefix of the dictionary keys corresponds to 'data transformation' and 'training'.

    :param run: Run object of AzureML trained model.

    :returns:    A dictionary as described in the docstring.
    """

    spec_field = "pipeline_spec"
    if spec_field not in run.properties:
        return {}
    spec_string = run.properties[spec_field]
    spec_dict = json.loads(spec_string)

    if "objects" not in spec_dict:
        # No hyper-params
        return {}
    hp_dicts = spec_dict["objects"]
    # after training there are two hyper-parameters dicts inside the run object:
    assert (
        len(hp_dicts) == 2
    ), "after training there are two hyper-parameters dicts inside the run object"
    result_dict = {}
    dict_keys = [
        ["data_trans_class_name", "data_trans_module", "data_trans_spec_class"],
        [
            "train_class_name",
            "train_module",
            "train_param_kwargs_C",
            "train_param_kwargs_class_weight",
            "train_spec_class",
        ],
    ]

    # creating hyper-params dict with key prefixes for each part:
    kwargs_prefix = "param_kwargs"
    for d, name, keys in zip(hp_dicts, ["data_trans", "train"], dict_keys):
        for key in keys:

            if kwargs_prefix in key:
                result_dict[key] = d[kwargs_prefix][
                    key.replace(f"{name}_{kwargs_prefix}_", "")
                ]
            else:
                result_dict[key] = d[key.replace(f"{name}_", "")]
            if not result_dict[key]:
                result_dict[key] = ""

    return result_dict


def submit_training_job(
    context: MLClientCtx,
    experiment: Experiment,
    compute_target: ComputeTarget,
    register_model_name: str,
    registered_dataset_name: str,
    automl_settings: dict,
    training_set: DataItem,
    label_column_name: str = '',
    save_n_models: int = 3,
    show_output: bool = True,
) -> None:
    """
    Submit training job to Azure AutoML and download trained model
    when completed. Uses previously registered dataset for training.

    :param context:                 MLRun context.
    :param experiment:              Azure experiment.
    :param compute_target:          Azure compute target.
    :param register_model_name:     Name of model to register in Azure.
    :param registered_dataset_name: Name of dataset registered in Azure ML.
    :param label_column_name:       Name of target column in dataset.
    :param automl_settings:         JSON string of all Azure AutoML settings.
    :param training_set:            Training set to log with model. For model
                                    monitoring integration.
    :param show_output:             Displaying Azure logs.
    :param save_n_models:           How many of the top performing models to log.
    """
    # Loading workspace if not provided:
    workspace = _load_workspace(context)

    # Setup experiment:
    context.logger.info("Setting up experiment parameters")
    dataset = Dataset.get_by_name(workspace, name=registered_dataset_name)

    # Get training set to log with model:
    feature_vector = None
    store_uri_prefix, _ = mlrun.datastore.parse_store_uri(training_set.artifact_url)
    if mlrun.utils.StorePrefix.FeatureVector == store_uri_prefix:
        feature_vector = training_set.meta.uri
        label_column_name = label_column_name or training_set.meta.status.label_column
        context.logger.info(f'label column name: {label_column_name}')
        training_set = f_store.get_offline_features(feature_vector).to_dataframe()
    else:
        training_set = training_set.as_df()

    automl_config = AutoMLConfig(
        compute_target=compute_target,
        training_data=dataset,
        verbosity=logging.INFO,
        label_column_name=label_column_name,
        **automl_settings,
    )

    # Run experiment on AzureML:
    context.logger.info("Submitting and running experiment")
    remote_run = experiment.submit(automl_config)
    remote_run.wait_for_completion(show_output=show_output)
    if show_output:
        # Azure log ending row:
        print(f"\n{'*' * 92}\n")
    # Get top N runs to log:
    top_runs = _get_top_n_runs(
        remote_run=remote_run,
        n=save_n_models,
        primary_metric=automl_settings["primary_metric"],
    )

    # Register, download, and log models:
    for i, run in enumerate(top_runs):
        # Register model:
        context.logger.info("Registering model")
        model = run.register_model(
            model_name=register_model_name, model_path="outputs/model.pkl"
        )
        context.logger.info(
            f"Registered model with name '{model.name}', id '{model.id}', version '{model.version}'"
        )

        # Download model locally:
        download_model(
            context=context,
            model_name=register_model_name,
            model_version=model.version,
            target_dir=f"./{model.version}",
        )

        metrics = {k.lower(): val for k, val in run.get_metrics().items()}
        del metrics["confusion_matrix"]
        del metrics["accuracy_table"]

        # Collect model hyper-parameters:
        model_hp_dict = _get_model_hp(run)
        with context.get_child_context(**model_hp_dict) as child:
            model_key = f"model_{i + 1}_{model_hp_dict['data_trans_class_name'].lower()}_{model_hp_dict['train_class_name'].lower()}"
            # Log model:
            context.logger.info(
                f"Logging {model_key} model to MLRun"
            )
            child.log_results(metrics)
            child.log_model(
                "model",
                db_key=model_key,
                artifact_path=context.artifact_subpath("models"),
                metrics=metrics,
                model_file=f"{model.version}/model.pkl",
                training_set=training_set,
                label_column=label_column_name,
                feature_vector=feature_vector,
                framework="AzureML",
                algorithm=model_hp_dict.get("train_class_name"),
            )
            if i == 0:
                # This also logs the model:
                child.mark_as_best()


def train(
    # MlRun
    context: MLClientCtx,
    dataset: DataItem,
    # Init experiment and compute
    experiment_name: str = "",
    cpu_cluster_name: str = "",
    vm_size: str = "STANDARD_D2_V2",
    max_nodes: int = 1,
    # Register dataset
    dataset_name: str = "",
    dataset_description: str = "",
    create_new_version: bool = False,
    label_column_name: str = "",
    # Submit training job
    register_model_name: str = "",
    save_n_models: int = 1,
    log_azure: bool = True,
    automl_settings: str = None,
) -> None:
    """
    Whole training flow for Azure AutoML. Registers dataset/feature vector,
    submits training job to Azure AutoML, and downloads trained model
    when completed.

    :param context:             MLRun context.

    :param dataset:             MLRun FeatureVector or dataset URI to upload. Will drop
                                index before uploading when it is a FeatureVector.

    :param experiment_name:     Name of experiment to create in Azure ML.
    :param cpu_cluster_name:    Name of Azure ML compute target. Created if does not exist.
    :param vm_size:             Azure machine type for compute target.
    :param max_nodes:           Maximum number of concurrent compute targets.

    :param dataset_name:        Name of Azure dataset to register.
    :param dataset_description: Description of Azure dataset to register.

    :param create_new_version:  Register Azure dataset as new version. Must be used when
                                modifying dataset schema.
    :param label_column_name:   Target column in dataset.

    :param register_model_name: Name of model to register in Azure.
    :param save_n_models:       How many of the top performing models to log.
    :param log_azure:           Displaying Azure logs.
    :param automl_settings:     JSON string of all Azure AutoML settings.
    """
    if not automl_settings:
        automl_settings = {
            "task": "classification",
            "debug_log": "automl_errors.log",
            # "experiment_exit_score": 0.9,
            "enable_early_stopping": False,
            "allowed_models": ["LogisticRegression", "SGD", "SVM"],
            "iterations": 3,
            "iteration_timeout_minutes": 2,
            "max_concurrent_iterations": 2,
            "max_cores_per_iteration": -1,
            "n_cross_validations": 5,
            "primary_metric": "accuracy",
            "featurization": "off",
            "model_explainability": False,
            "enable_voting_ensemble": False,
            "enable_stack_ensemble": False,
        }

    # Init experiment and compute
    workspace, experiment = _init_experiment(
        context=context, experiment_name=experiment_name
    )

    compute_target = init_compute(
        context=context,
        cpu_cluster_name=cpu_cluster_name,
        vm_size=vm_size,
        max_nodes=max_nodes,
    )

    # Register dataset
    register_dataset(
        context=context,
        dataset_name=dataset_name,
        dataset_description=dataset_description,
        data=dataset,
        create_new_version=create_new_version,
    )

    # Submit training job
    submit_training_job(
        context,
        experiment=experiment,
        compute_target=compute_target,
        register_model_name=register_model_name,
        registered_dataset_name=dataset_name,
        label_column_name=label_column_name,
        automl_settings=automl_settings,
        training_set=dataset,
        show_output=log_azure,
        save_n_models=save_n_models,
    )
