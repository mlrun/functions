import os
import json
import logging
from typing import Tuple, List

from mlrun import MLClientCtx, DataItem, get_dataitem
import mlrun.feature_store as f_store
from mlrun.api.schemas import ObjectKind
from mlrun.datastore.targets import kind_to_driver

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


def _load_workspace(context: MLClientCtx) -> Workspace:
    """
    Loading AzureML Workspace using Azure secrets.

    :param context: MLRun context.
    :returns:       AzureML Workspace
    """

    context.logger.info("Loading AzureML Workspace")
    # Azure service authentication:
    service_authentication = ServicePrincipalAuthentication(
        tenant_id=context.get_secret("AZURE_TENANT_ID"),
        service_principal_id=context.get_secret("AZURE_SERVICE_PRINCIPAL_ID"),
        service_principal_password=context.get_secret(
            "AZURE_SERVICE_PRINCIPAL_PASSWORD"
        ),
    )

    # Loading Azure workspace:
    workspace = Workspace(
        subscription_id=context.get_secret("AZURE_SUBSCRIPTION_ID"),
        resource_group=context.get_secret("AZURE_RESOURCE_GROUP"),
        workspace_name=context.get_secret("AZURE_WORKSPACE_NAME"),
        auth=service_authentication,
    )
    return workspace


def init_experiment(
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

    context.logger.info("Initializing AzureML experiment")
    # Creating experiment:
    experiment = Experiment(workspace, experiment_name)

    # Set Azure storage connection environment variable from secret:
    assert context.get_secret(
        "AZURE_STORAGE_CONNECTION_STRING"
    ), "AZURE_STORAGE_CONNECTION_STRING secret not set"

    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = context.get_secret(
        "AZURE_STORAGE_CONNECTION_STRING"
    )

    return workspace, experiment


def init_compute(
    context: MLClientCtx,
    cpu_cluster_name: str,
    workspace: Workspace = None,
    vm_size: str = "STANDARD_D2_V2",
    max_nodes: int = 1,
) -> ComputeTarget:
    """
    Initialize Azure ML compute target to run experiment. Checks for
    existing compute target and creates new if doesn't exist.

    :param context:          MLRun context.
    :param cpu_cluster_name: Name of Azure ML compute target. Created if does not exist.
    :param workspace:        Azure Workspace.
    :param vm_size:          Azure machine type for compute target.
    :param max_nodes:        Maximum number of concurrent compute targets.
    :returns:                Azure ML Compute Target.
    """
    # Loading workspace if not provided:
    if not workspace:
        workspace = _load_workspace(context)

    context.logger.info("Initializing AzureML compute target")

    # Verify that cluster does not exist already:
    try:
        compute_target = ComputeTarget(workspace=workspace, name=cpu_cluster_name)
        context.logger.info("Found existing cluster, use it.")
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
    workspace: Workspace = None,
    drop_columns: list = None,
    create_new_version: bool = False,
) -> DataItem:
    """
    Register dataset object (can be also an Iguazio FeatureVector) in Azure ML.
    Uploads parquet file to Azure blob storage and registers
    that file as a dataset in Azure ML.

    :param context:               MLRun context.
    :param workspace:             Azure workspace.
    :param dataset_name:          Name of Azure dataset to register.
    :param dataset_description:   Description of Azure dataset to register.
    :param data:                  Iguazio FeatureVector or dataset object to upload. Will drop
                                  index before uploading when it is a FeatureVector.
    :param drop_columns:          List of columns to drop from uploaded dataset.
    :param create_new_version:    Register Azure dataset as new version. Must be used when
                                  modifying dataset schema.

    :returns:                     dataset or feature vector
    """

    data_type = data.suffix
    # Connect to AzureML experiment and datastore:
    context.logger.info("Connecting to AzureML experiment default datastore")

    # Loading workspace if not provided:
    if not workspace:
        workspace = _load_workspace(context)
    datastore = workspace.get_default_datastore()

    # Azure blob path (default datastore for workspace):
    blob_path = f"az://{datastore.container_name}/{dataset_name}{data_type}"

    # Retrieve data source as dataframe:
    if data.meta and data.meta.kind == ObjectKind.feature_vector:
        # FeatureVector case:
        context.logger.info(
            "Retrieving feature vector and uploading to Azure blob storage"
        )
        target = kind_to_driver[data_type](path=blob_path)
        f_store.get_offline_features(data, target=target)
    else:
        # DataItem case:
        context.logger.info("Retrieving dataset and uploading to Azure blob storage")
        data_in_bytes = data.get()
        get_dataitem(blob_path).put(data_in_bytes)

    # Register dataset in AzureML:
    context.logger.info("Registering dataset in Azure ML")
    ds = Dataset.File.from_files(path=[(datastore, f'{dataset_name}{data_type}')])

    ds.register(
        workspace=workspace,
        name=dataset_name,
        description=dataset_description,
        create_new_version=create_new_version,
    )

    # Output registered dataset name in Azure:
    context.log_artifact("registered_dataset_name", dataset_name)
    context.log_artifact("blob_path", blob_path)

    # Return data source:
    return data


def download_model(
    context: MLClientCtx,
    model_name: str,
    model_version: int,
    workspace: Workspace = None,
    target_dir: str = ".",
) -> None:
    """
    Download trained model from Azure ML to local filesystem.

    :param context:       MLRun context.
    :param workspace:     Azure ML Workspace.
    :param model_name:    Name of trained and registered model.
    :param model_version: Version of model to download.
    :param target_dir:    Target directory to download model.
    """
    # Loading workspace if not provided:
    if not workspace:
        workspace = _load_workspace(context)

    context.logger.info("Downloading model")
    model = Model(workspace, model_name, version=model_version)
    model.download(target_dir=target_dir, exist_ok=True)


def upload_model(
    context: MLClientCtx,
    model_name: str,
    model_path: str,
    workspace: Workspace = None,
    model_description: str = None,
    model_tags: dict = None,
) -> None:
    """
    Upload pre-trained model from local filesystem to Azure ML.
    :param context:           MLRun context.
    :param workspace:         Azure ML Workspace.
    :param model_name:        Name of trained and registered model.
    :param model_path:        Path to file on local filesystem.
    :param model_description: Description of models.
    :param model_tags:        KV pairs of model tags.
    """
    # Loading workspace if not provided:
    if not workspace:
        workspace = _load_workspace(context)

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
        if "setup" not in run.id
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

    # Creating temp json file because ast.literal_eval fails:
    filename = "hp_tmp.json"
    tmp_file = open(
        f"./{filename}", 'w'
    )
    # Writing complex dict to file:
    tmp_file.write(spec_string)
    tmp_file.close()
    # Reading for loading:
    tmp_file = open(
        f"./{filename}", 'r'
    )
    # Creating dict:
    spec_dict = json.load(tmp_file)
    tmp_file.close()
    # Deleting file:
    os.remove(os.path.abspath(f"./{filename}"))

    if "objects" not in spec_dict:
        # No hyper-params
        return {}
    hp_dicts = spec_dict["objects"]
    # after training there are two hyper-parameters dicts inside the run object:
    assert len(hp_dicts) == 2
    result_dict = {}

    # creating hyper-params dict with key prefixes for each part:
    for d, name in zip(hp_dicts, ["data_transform", "train"]):
        for key in d.keys():
            if d[key]:
                result_dict[f"{name}_{key}"] = d[key]

    return result_dict


def submit_training_job(
    context: MLClientCtx,
    experiment: Experiment,
    compute_target: ComputeTarget,
    register_model_name: str,
    registered_dataset_name: str,
    label_column_name: str,
    automl_settings: str,
    training_set: DataItem,
    workspace: Workspace = None,
    save_n_models: int = 3,
    show_output: bool = True,
) -> None:
    """
    Submit training job to Azure AutoML and download trained model
    when completed. Uses previously registered dataset for training.

    :param context:                 MLRun context.
    :param workspace:               Azure workspace.
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
    if not workspace:
        workspace = _load_workspace(context)

    # Setup experiment:
    context.logger.info("Setting up experiment parameters")
    dataset = Dataset.get_by_name(workspace, name=registered_dataset_name)
    automl_settings = json.loads(automl_settings)
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

    # Get top N runs to log:
    top_runs = _get_top_n_runs(
        remote_run=remote_run,
        n=save_n_models,
        primary_metric=automl_settings["primary_metric"],
    )

    # Get training set to log with model:
    training_set = training_set.as_df().drop(label_column_name, axis=1)

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
            workspace=workspace,
            model_name=register_model_name,
            model_version=model.version,
            target_dir=f"./{model.version}",
        )

        # Log model metrics:
        metrics = run.get_metrics()
        del metrics["confusion_matrix"]
        del metrics["accuracy_table"]

        # Collect model hyper-parameters:
        model_hp_dict = _get_model_hp(run)
        with context.get_child_context(**model_hp_dict) as child:
            if i == 0:
                child.mark_as_best()
            child.log_model(f'{i}',
                            parameters=child.parameters)

        # Log model:
        context.logger.info("Logging model to MLRun")
        context.log_model(
            f"model_{i}",
            artifact_path=context.artifact_subpath(f"{model.version}"),
            metrics=metrics,
            model_file=f"{model.version}/model.pkl",
            training_set=training_set,
        )


def automl_train(
    # MlRun
    context: MLClientCtx,
    training_data_uri: DataItem,
    # Init experiment and compute
    experiment_name: str = "",
    cpu_cluster_name: str = "",
    vm_size: str = "STANDARD_D2_V2",
    max_nodes: int = 1,
    # Register dataset
    dataset_name: str = "",
    dataset_description: str = "",
    feature_vector_entity="",
    drop_columns: list = None,
    create_new_version: bool = False,
    label_column_name: str = "",
    # Submit training job
    register_model_name: str = "",
    save_n_models: int = 3,
    log_azure: bool = True,
    automl_settings: str = None,
) -> None:
    """
    Whole training flow for Azure AutoML. Registers dataset/feature vector,
    submits training job to Azure AutoML, and downloads trained model
    when completed.

    :param context:               MLRun context.

    :param experiment_name:       Name of experiment to create in Azure ML.
    :param cpu_cluster_name:      Name of Azure ML compute target. Created if does not exist.
    :param vm_size:               Azure machine type for compute target.
    :param max_nodes:             Maximum number of concurrent compute targets.

    :param dataset_name:          Name of Azure dataset to register.
    :param dataset_description:   Description of Azure dataset to register.
    :param feature_vector_entity: Entity (primary key) of feature vector. Dropped when registering
                                  dataset in Azure.
    :param training_data_uri:     Iguazio FeatureVector or dataset URI to upload. Will drop
                                  index before uploading when it is a FeatureVector.
    :param drop_columns:          List of columns to drop from uploaded dataset. Defaults to
                                  feature_vector_entity.
    :param create_new_version:    Register Azure dataset as new version. Must be used when
                                  modifying dataset schema.
    :param label_column_name:     Target column in dataset.

    :param register_model_name:   Name of model to register in Azure.
    :param save_n_models:         How many of the top performing models to log.
    :param log_azure:             Displaying Azure logs.
    :param automl_settings:       JSON string of all Azure AutoML settings.
    """
    if not automl_settings:
        automl_settings = json.dumps(
            {
                "task": "classification",
                "debug_log": "automl_errors.log",
                # "experiment_exit_score": 0.9,
                "enable_early_stopping": False,
                "allowed_models": ["LogisticRegression", "SGD", "SVM"],
                "iterations": 2,
                "iteration_timeout_minutes": 2,
                "max_concurrent_iterations": 2,
                "max_cores_per_iteration": -1,
                "n_cross_validations": 5,
                "primary_metric": "accuracy",
                "featurization": "auto",
                "model_explainability": False,
                "enable_voting_ensemble": False,
                "enable_stack_ensemble": False,
            }
        )

    # Init experiment and compute
    workspace, experiment = init_experiment(
        context=context, experiment_name=experiment_name
    )

    compute_target = init_compute(
        context=context,
        workspace=workspace,
        cpu_cluster_name=cpu_cluster_name,
        vm_size=vm_size,
        max_nodes=max_nodes,
    )

    # Register dataset
    training_set = register_dataset(
        context=context,
        workspace=workspace,
        dataset_name=dataset_name,
        dataset_description=dataset_description,
        data=training_data_uri,
        drop_columns=drop_columns or [feature_vector_entity],
        create_new_version=create_new_version,
    )

    # Submit training job
    submit_training_job(
        context,
        workspace=workspace,
        experiment=experiment,
        compute_target=compute_target,
        register_model_name=register_model_name,
        registered_dataset_name=dataset_name,
        label_column_name=label_column_name,
        automl_settings=automl_settings,
        training_set=training_set,
        show_output=log_azure,
        save_n_models=save_n_models,
    )
