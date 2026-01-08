"""
MLRun to LangChain integration - a tracer that converts LangChain Run objects into MLRun monitoring events and
publishes them to a V3IO stream via MLRun endpoint monitoring format.
"""

import copy
import importlib
import orjson
import os
import socket
from uuid import UUID
import threading
from contextlib import contextmanager
from contextvars import ContextVar
import datetime
from typing import Any, Callable, Generator, Optional

import v3io
from langchain_core.tracers import BaseTracer, Run
from langchain_core.tracers.context import register_configure_hook

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from uuid_utils import uuid7

import mlrun
from mlrun.runtimes import RemoteRuntime
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase, ModelMonitoringApplicationMetric,
    ModelMonitoringApplicationResult, MonitoringApplicationContext,
)
import mlrun.common.schemas.model_monitoring.constants as mm_constants

#: Environment variable name to use MLRun monitoring tracer via LangChain global tracing system:
mlrun_monitoring_env_var = "MLRUN_MONITORING_ENABLED"


class _MLRunEndPointClient:
    """
    An MLRun model endpoint monitoring client to connect and send events on a V3IO stream.
    """

    def __init__(
        self,
        monitoring_stream_path: str,
        monitoring_container: str,
        model_endpoint_name: str,
        model_endpoint_uid: str,
        serving_function: str | RemoteRuntime,
        serving_function_tag: str | None = None,
        project: str | mlrun.projects.MlrunProject = None,
    ):
        """
        Initialize an MLRun model endpoint monitoring client.

        :param monitoring_stream_path: V3IO stream path.
        :param monitoring_container: V3IO container name.
        :param model_endpoint_name: The monitoring endpoint related model name.
        :param model_endpoint_uid: Model endpoint unique identifier.
        :param serving_function: Serving function name or ``RemoteRuntime`` object.
        :param serving_function_tag: Optional function tag (defaults to 'latest').
        :param project: Project name or ``MlrunProject``. If ``None``, uses the current project.

        raise: MLRunInvalidArgumentError: If there is no current active project and no `project` argument was provided.
        """
        # Store the provided info:
        self._monitoring_stream_path = monitoring_stream_path
        self._monitoring_container = monitoring_container
        self._model_endpoint_name = model_endpoint_name
        self._model_endpoint_uid = model_endpoint_uid

        # Load project:
        if project is None:
            try:
                self._project_name = mlrun.get_current_project(silent=False).name
            except mlrun.errors.MLRunInvalidArgumentError:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "There is no current active project. Either use `mlrun.get_or_create_project` prior to "
                    "initializing the monitoring tracer or pass a project name to load. You can also set the "
                    "environment variable: 'MLRUN_MONITORING_PROJECT'."
                )
        elif isinstance(project, str):
            self._project_name = project
        else:
            self._project_name = project.name

        # Load function:
        if isinstance(serving_function, str):
            self._serving_function_name = serving_function
            self._serving_function_tag = serving_function_tag or "latest"
        else:
            self._serving_function_name = serving_function.metadata.name
            self._serving_function_tag = (
                serving_function_tag or serving_function.metadata.tag
            )

        # Initialize a V3IO client:
        self._v3io_client = v3io.Client()

        # Prepare the sample:
        self._event_sample = {
            "class": "CustomStream",
            "worker": "0",
            "model": self._model_endpoint_name,
            "host": socket.gethostname(),
            "function_uri": f"{self._project_name}/{self._serving_function_name}:{self._serving_function_tag}",
            "endpoint_id": self._model_endpoint_uid,
            "sampling_percentage": 100,
            "request": {"inputs": [], "background_task_state": "succeeded"},
            "op": "infer",
            "resp": {
                "id": None,
                "model_name": self._model_endpoint_name,
                "outputs": [],
                "timestamp": None,
                "model_endpoint_uid": self._model_endpoint_uid,
            },
            "when": None,
            "microsec": 496,
            "effective_sample_count": 1,
        }

    def monitor(
        self,
        event_id: str,
        label: str,
        input_data: dict,
        output_data: dict,
        request_timestamp: str,
        response_timestamp: str,
    ):
        """
        Monitor the provided event, sending it to the model endpoint monitoring stream.

        :param event_id: Unique event identifier used as the monitored record id.
        :param label: Label for the run/event.
        :param input_data: Serialized input data for the run.
        :param output_data: Serialized output data for the run.
        :param request_timestamp: Request/start timestamp in the format of '%Y-%m-%d %H:%M:%S%z'.
        :param response_timestamp: Response/end timestamp in the format of '%Y-%m-%d %H:%M:%S%z'.
        """
        # Copy the sample:
        event = copy.deepcopy(self._event_sample)

        # Edit event with given parameters:
        event["when"] = request_timestamp
        event["request"]["inputs"].append(orjson.dumps({"label": label, "input": input_data}).decode('utf-8'))
        event["resp"]["timestamp"] = response_timestamp
        event["resp"]["outputs"].append(orjson.dumps(output_data).decode('utf-8'))
        event["resp"]["id"] = event_id

        # Push to stream:
        self._v3io_client.stream.put_records(
            container=self._monitoring_container,
            stream_path=self._monitoring_stream_path,
            records=[{"data": orjson.dumps(event).decode('utf-8')}],
        )


class MLRunTracerClientSettings(BaseSettings):
    """
    MLRun tracer monitoring client configurations. These are mandatory arguments for allowing MLRun to send monitoring
    events to a specific model endpoint stream.
    """

    stream_path: str = ...
    """
    The V3IO stream path to send the events to.
    """

    container: str = ...
    """
    The V3IO stream container.
    """

    model_endpoint_name: str = ...
    """
    The model endpoint name.
    """

    model_endpoint_uid: str = ...
    """
    The model endpoint UID.
    """

    serving_function: str = ...
    """
    The serving function name.
    """

    serving_function_tag: str | None = None
    """
    The serving function tag. If not set, it will be 'latest' by default.
    """

    project: str | None = None
    """
    The MLRun project name related to the serving function and model endpoint.
    """

    #: Pydantic model configuration to set the environment variable prefix.
    model_config = SettingsConfigDict(env_prefix="MLRUN_TRACER_CLIENT_")


class MLRunTracerMonitorSettings(BaseSettings):
    """
    MLRun tracer monitoring configurations. These are optional arguments to customize the LangChain runs summarization
    into monitorable MLRun endpoint events. If needed, a custom summarization can be passed.
    """

    label: str = "default"
    """
    Label to use for all monitored runs. Can be used to differentiate between different monitored sources on the same 
    endpoint.
    """

    tags_filter: list[str] | None = None
    """
    Filter runs by tags. Only runs with at least one tag in this list will be monitored.
    If None, no tag-based filtering is applied and runs with any tags are considered.
    Default: None.
    """

    run_types_filter: list[str] | None = None
    """
    Filter runs by run types (e.g. "chain", "llm", "chat", "tool").
    Only runs whose `run_type` appears in this list will be monitored.
    If None, no run-type filtering is applied.
    Default: None.
    """

    names_filter: list[str] | None = None
    """
    Filter runs by class/name. Only runs whose `name` appears in this list will be monitored.
    If None, no name-based filtering is applied.
    Default: None.
    """

    include_full_run: bool = False
    """
    If True, include the complete serialized run dict (the output of `run._get_dicts_safe()`)
    in the event outputs under the key `full_run`. Useful for debugging or when consumers need
    the raw run payload. Default: False.
    """

    include_errors: bool = True
    """
    If True, include run error information in the outputs under the `error` key.
    If False, runs that contain an error may be skipped by the summarizer filters.
    Default: True.
    """

    include_metadata: bool = True
    """
    If True, include run metadata (environment, tool metadata, etc.) in the inputs under
    the `metadata` key. Default: True.
    """

    include_latency: bool = True
    """
    If True, include latency information in the outputs under the `latency` key.
    Default: True.
    """

    root_run_only: bool = False
    """
    If True, only the root/top-level run will be monitored and any child runs will be
    ignored/removed from monitoring. Use when only the top-level run should produce events.
    Default: False.
    """

    split_runs: bool = False
    """
    If True, child runs are emitted as separate monitoring events (each run summarized and
    sent individually). If False, child runs are nested inside the parent/root run event under
    `child_runs`. Default: False.
    """

    run_summarizer_function: (
        str
        | Callable[
            [Run, Optional[BaseSettings]],
            Generator[tuple[dict, dict] | None, None, None],
        ]
        | None
    ) = None
    """
    A function to summarize a `Run` object into a tuple of inputs and outputs. Can be passed directly or via a full 
    module path ("a.b.c.my_summarizer" will be imported as `from a.b.c import my_summarizer`).

    A summarizer is a function that will be used to process a run into monitoring events. The function is expected to be 
    of type:
    `Callable[[Run, Optional[BaseSettings]], Generator[tuple[dict, dict] | None, None, None]]`, meaning 
    get a run object and optionally a settings object and return a generator yielding tuples of serialized dictionaries, 
    the (inputs, outputs) to send to MLRun monitoring as events or `None` to skip monitoring this run.
    """

    run_summarizer_settings: str | BaseSettings | None = None
    """
    Settings to pass to the run summarizer function. Can be passed directly or via a full module path to be imported
    and initialized. If the summarizer function does not require settings, this can be left as None.
    """

    debug: bool = False
    """
    If True, disable sending events to MLRun/V3IO and instead route events to `debug_target_list`
    or print them as JSON to stdout. Useful for unit tests and local debugging. Default: False.
    """

    debug_target_list: list[dict] | bool = False
    """
    Optional list to which debug events will be appended when `debug` is True.
    If set, each generated event dict will be appended to this list. If not set and `debug` is True,
    events will be printed to stdout as JSON. Default: False.
    """

    #: Pydantic model configuration to set the environment variable prefix.
    model_config = SettingsConfigDict(env_prefix="MLRUN_TRACER_MONITOR_")

    @field_validator('debug_target_list', mode='before')
    @classmethod
    def convert_bool_to_list(cls, v):
        """
        Convert a boolean `True` value to an empty list for `debug_target_list`.

        :param v: The value to validate.

        :returns: An empty list if `v` is True, otherwise the original value.
        """
        if v is True:
            return []
        return v


class MLRunTracerSettings(BaseSettings):
    """
    MLRun tracer settings to configure the tracer. The settings are split into two groups:

    * `client`: settings required to connect and send events to the MLRun/V3IO monitoring stream.
    * `monitor`: settings controlling which LangChain runs are summarized and sent and how.
    """

    client: MLRunTracerClientSettings = Field(default_factory=MLRunTracerClientSettings)
    """
    Client configuration group (``MLRunTracerClientSettings``).

    Contains the mandatory connection and endpoint information required to publish monitoring
    events. Values may be supplied programmatically or via environment variables prefixed with
    `MLRUN_TRACER_CLIENT_`. See more at ``MLRunTracerClientSettings``.
    """

    monitor: MLRunTracerMonitorSettings = Field(default_factory=MLRunTracerMonitorSettings)
    """
    Monitoring configuration group (``MLRunTracerMonitorSettings``).

    Controls what runs are captured, how they are summarized (including custom summarizer import
    options), whether child runs are split or nested, and debug behavior. Values may be supplied
    programmatically or via environment variables prefixed with `MLRUN_TRACER_MONITOR_`. 
    See more at ``MLRunTracerMonitorSettings``.
    """

    #: Pydantic model configuration to set the environment variable prefix.
    model_config = SettingsConfigDict(env_prefix="MLRUN_TRACER_")


class MLRunTracer(BaseTracer):
    """
    MLRun tracer for LangChain runs allowing monitoring LangChain and LangGraph in production using MLRun's monitoring.

    There are two usage modes for the MLRun tracer following LangChain tracing best practices:

    1. **Manual Mode** - Using the ``mlrun_monitoring`` context manager::

        from mlrun_tracer import mlrun_monitoring

        with mlrun_monitoring(...) as tracer:
            # LangChain code here.
            pass

    2. **Auto Mode** - Setting the `MLRUN_MONITORING_ENABLED="1"` environment variable::

        import mlrun_integration.tracer

        # All LangChain code will be automatically traced and monitored.
        pass

    To control how runs are being summarized into the events being monitored, the ``MLRunTracerSettings`` can be set.
    As it is a Pydantic ``BaseSettings`` class, it can be done in two ways:

    1. Initializing the settings classes and passing them to the context manager::

        from mlrun_tracer import (
            mlrun_monitoring,
            MLRunTracerSettings,
            MLRunTracerClientSettings,
            MLRunTracerMonitorSettings,
        )

        my_settings = MLRunTracerSettings(
            client=MLRunTracerClientSettings(),
            monitor=MLRunTracerMonitorSettings(root_run_only=True),
        )

        with mlrun_monitoring(settings=my_settings) as tracer:
            # LangChain code here.
            pass

    2. Or via environment variables following the prefix 'MLRUN_TRACER_CLIENT_' for client settings and
       'MLRUN_TRACER_MONITOR_' for monitoring settings.
    """

    #: A singleton tracer for when using the tracer via environment variable to activate global tracing.
    _singleton_tracer: "MLRunTracer | None" = None
    #: A thread lock for initializing the tracer singleton safely.
    _lock = threading.Lock()
    #: A boolean flag to know whether the singleton was initialized.
    _initialized = False

    def __new__(cls, *args, **kwargs) -> "MLRunTracer":
        """
        Create or return an ``MLRunTracer`` instance.

        When ``MLRUN_MONITORING_ENABLED`` is not set to ``"1"``, a normal instance is returned.
        When the env var is ``"1"``, a process-wide singleton is returned. Creation is thread-safe.

        :returns: MLRunTracer instance (singleton if 'auto' mode is active).
        """
        # Check if needed to use a singleton as the user is using the MLRun tracer by setting the environment variable
        # and not manually (via context manager):
        if not cls._check_for_env_var_usage():
            return super(MLRunTracer, cls).__new__(cls)

        # Check if the singleton is set:
        if cls._singleton_tracer is None:
            # Acquire lock to initialize the singleton:
            with cls._lock:
                # Double-check after acquiring lock:
                if cls._singleton_tracer is None:
                    cls._singleton_tracer = super(MLRunTracer, cls).__new__(cls)

        return cls._singleton_tracer

    def __init__(self, settings: MLRunTracerSettings = None, **kwargs):
        """
        Initialize the tracer.

        :param settings: Settings to use for the tracer. If not passed, defaults are used and environment variables are
            applied per Pydantic settings behavior.
        :param kwargs: Passed to the base initializer.
        """
        # Proceed with initialization only if singleton mode is not required or the singleton was not initialized:
        if self._check_for_env_var_usage() and self._initialized:
            return

        # Call the base tracer init:
        super().__init__(**kwargs)

        # Set a UID for this instance:
        self._uid = uuid7()

        # Set the settings:
        self._settings = settings or MLRunTracerSettings()
        self._client_settings = self._settings.client
        self._monitor_settings = self._settings.monitor

        # Initialize the MLRun endpoint client:
        self._mlrun_client = (
            _MLRunEndPointClient(
                monitoring_stream_path=self._client_settings.stream_path,
                monitoring_container=self._client_settings.container,
                model_endpoint_name=self._client_settings.model_endpoint_name,
                model_endpoint_uid=self._client_settings.model_endpoint_uid,
                serving_function=self._client_settings.serving_function,
                serving_function_tag=self._client_settings.serving_function_tag,
                project=self._client_settings.project,
            )
            if not self._monitor_settings.debug
            else None
        )

        # In case the user passed a custom summarizer, import it:
        self._custom_run_summarizer_function: (
            Callable[
                [Run, Optional[BaseSettings]],
                Generator[tuple[dict, dict] | None, None, None],
            ]
            | None
        ) = None
        self._custom_run_summarizer_settings: BaseSettings | None = None
        self._import_custom_run_summarizer()

        # Mark the initialization flag (for the singleton case):
        self._initialized = True

    @property
    def settings(self) -> MLRunTracerSettings:
        """
        Access the effective settings.

        :returns: The settings used by this tracer.
        """
        return self._settings

    def _import_custom_run_summarizer(self):
        """
        Import or assign a custom run summarizer (and its custom settings) if configured.
        """
        # If the user did not pass a run summarizer function, return:
        if not self._monitor_settings.run_summarizer_function:
            return

        # Check if the function needs to be imported:
        if isinstance(self._monitor_settings.run_summarizer_function, str):
            self._custom_run_summarizer_function = self._import_from_module_path(
                module_path=self._monitor_settings.run_summarizer_function
            )
        else:
            self._custom_run_summarizer_function = (
                self._monitor_settings.run_summarizer_function
            )

        # Check if the user passed settings as well:
        if self._monitor_settings.run_summarizer_settings:
            # Check if the settings need to be imported:
            if isinstance(self._monitor_settings.run_summarizer_settings, str):
                self._custom_run_summarizer_settings = self._import_from_module_path(
                    module_path=self._monitor_settings.run_summarizer_settings
                )()
            else:
                self._custom_run_summarizer_settings = (
                    self._monitor_settings.run_summarizer_settings
                )

    def _persist_run(self, run: Run, level: int = 0) -> None:
        """
        Summarize the run (and its children) into MLRun monitoring events.

        Note: This will use the MLRun tracer's default summarization that can be configured via
        ``MLRunTracerMonitorSettings``, unless a custom summarizer was provided (via the same settings).

        :param run: LangChain run object to process holding all the nested tree of runs.
        :param level: The nesting level of the run (0 for root runs, incremented for child runs).
        """
        # Serialize the run:
        serialized_run = self._serialize_run(
            run=run,
            include_child_runs=not (self._settings.monitor.root_run_only or self._settings.monitor.split_runs)
        )

        # Check for a user custom run summarizer function:
        if self._custom_run_summarizer_function:
            for summarized_run in self._custom_run_summarizer_function(
                run, self._custom_run_summarizer_settings
            ):
                if summarized_run:
                    inputs, outputs = summarized_run
                    self._send_run_event(
                        event_id=serialized_run["id"],
                        inputs=inputs,
                        outputs=outputs,
                        start_time=run.start_time,
                        end_time=run.end_time,
                    )
            return

        # Check how to deal with the child runs, monitor them in separate events or as a single event:
        if self._monitor_settings.split_runs and not self._settings.monitor.root_run_only:
            # Monitor as separate events:
            for child_run in run.child_runs:
                self._persist_run(run=child_run, level=level + 1)
            summarized_run = self._summarize_run(serialized_run=serialized_run, include_children=False)
            if summarized_run:
                inputs, outputs = summarized_run
                inputs["child_level"] = level
                self._send_run_event(
                    event_id=serialized_run["id"],
                    inputs=inputs,
                    outputs=outputs,
                    start_time=run.start_time,
                    end_time=run.end_time,
                )
            return

        # Monitor the root event (include child runs if `root_run_only` is False):
        summarized_run = self._summarize_run(
            serialized_run=serialized_run,
            include_children=not self._monitor_settings.root_run_only
        )
        if not summarized_run:
            return
        inputs, outputs = summarized_run
        inputs["child_level"] = level
        self._send_run_event(
            event_id=serialized_run["id"],
            inputs=inputs,
            outputs=outputs,
            start_time=run.start_time,
            end_time=run.end_time,
        )


    def _serialize_run(self, run: Run, include_child_runs: bool) -> dict:
        """
        Serialize a LangChain run into a dictionary.

        :param run: The run to serialize.
        :param include_child_runs: Whether to include child runs in the serialization.

        :returns: The serialized run dictionary.
        """
        # In LangChain 1.2.3+, the Run model uses Pydantic v2 with child_runs marked as Field(exclude=True), so we
        # must manually serialize child runs. Still excluding manually for future compatibility. In previous
        # LangChain versions, Run was Pydantic v1, so we use dict.
        serialized_run = (
            run.model_dump(exclude={"child_runs"})
            if hasattr(run, "model_dump")
            else run.dict(exclude={"child_runs"})
        )

        # Manually serialize child runs if needed:
        if include_child_runs and run.child_runs:
            serialized_run["child_runs"] = [
                self._serialize_run(child_run, include_child_runs=True)
                for child_run in run.child_runs
            ]

        return orjson.loads(orjson.dumps(serialized_run, default=self._serialize_default))

    def _serialize_default(self, obj: Any):
        """
        Default serializer for objects present in LangChain run that are not serializable by default JSON encoder. It
        includes handling Pydantic v1 and v2 models, UUIDs, and datetimes.

        :param obj: The object to serialize.

        :returns: The serialized object.
        """
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            return orjson.loads(orjson.dumps(obj.model_dump(), default=self._serialize_default))
        if hasattr(obj, "dict"):
            return orjson.loads(orjson.dumps(obj.dict(), default=self._serialize_default))
        return str(obj)

    def _filter_by_tags(self, serialized_run: dict) -> bool:
        """
        Apply tag-based filtering.

        :param serialized_run: Serialized run dictionary.

        :returns: True if the run passes tag filters or if no tag filter is configured.
        """
        # Check if the user enabled filtering by tags:
        if not self._monitor_settings.tags_filter:
            return True

        # Filter the run:
        return not set(self._monitor_settings.tags_filter).isdisjoint(
            serialized_run["tags"]
        )

    def _filter_by_run_types(self, serialized_run: dict) -> bool:
        """
        Apply run-type filtering.

        :param serialized_run: Serialized run dictionary.

        :returns: True if the run's ``run_type`` is allowed or if no run-type filter is configured.
        """
        # Check if the user enabled filtering by run types:
        if not self._monitor_settings.run_types_filter:
            return True

        # Filter the run:
        return serialized_run["run_type"] in self._monitor_settings.run_types_filter

    def _filter_by_names(self, serialized_run: dict) -> bool:
        """
        Apply class/name filtering.

        :param serialized_run: Serialized run dictionary.

        :returns: True if the run's ``name`` is allowed or if no name filter is configured.
        """
        # Check if the user enabled filtering by class names:
        if not self._monitor_settings.names_filter:
            return True

        # Filter the run:
        return serialized_run["name"] in self._monitor_settings.names_filter

    def _get_run_inputs(self, serialized_run: dict) -> dict[str, Any]:
        """
        Build the inputs dictionary for a monitoring event.

        :param serialized_run: Serialized run dictionary.

        :returns: A dictionary containing inputs, run metadata and (optionally) additional metadata.
        """
        inputs = {
            "inputs": serialized_run["inputs"],
            "run_type": serialized_run["run_type"],
            "run_name": serialized_run["name"],
            "tags": serialized_run["tags"],
            "run_id": serialized_run["id"],
            "start_timestamp": serialized_run["start_time"],
        }
        if "parent_run_id" in serialized_run:
            # Parent run ID is excluded when child runs are joined in the same event. When child runs are split, it is
            # included and can be used to reconstruct the run tree if needed.
            inputs = {**inputs, "parent_run_id": serialized_run["parent_run_id"]}
        if self._monitor_settings.include_metadata and "metadata" in serialized_run:
            inputs = {**inputs, "metadata": serialized_run["metadata"]}

        return inputs

    def _get_run_outputs(self, serialized_run: dict) -> dict[str, Any]:
        """
        Build the outputs dictionary for a monitoring event.

        :param serialized_run: Serialized run dictionary.

        :returns: A dictionary with outputs and optional other collected info depending on monitor settings.
        """
        outputs = {"outputs": serialized_run["outputs"], "end_timestamp": serialized_run["end_time"]}
        if self._monitor_settings.include_latency and "latency" in serialized_run:
            outputs = {**outputs, "latency": serialized_run["latency"]}
        if self._monitor_settings.include_errors:
            outputs = {**outputs, "error": serialized_run["error"]}
        if self._monitor_settings.include_full_run:
            outputs = {**outputs, "full_run": serialized_run}

        return outputs

    def _summarize_run(self, serialized_run: dict, include_children: bool) -> tuple[dict, dict] | None:
        """
        Summarize a single run into (inputs, outputs) if it passes filters.

        :param serialized_run: Serialized run dictionary.
        :param include_children: Whether to include child runs.

        :returns: The summarized run (inputs, outputs) tuple if the run should be monitored, otherwise ``None``.
        """
        # Pass filters:
        if not (
            self._filter_by_tags(serialized_run=serialized_run)
            and self._filter_by_run_types(serialized_run=serialized_run)
            and self._filter_by_names(serialized_run=serialized_run)
        ):
            return None

        # Check if needed to include errors:
        if serialized_run["error"] and not self._monitor_settings.include_errors:
            return None

        # Prepare the inputs and outputs:
        inputs = self._get_run_inputs(serialized_run=serialized_run)
        outputs = self._get_run_outputs(serialized_run=serialized_run)

        # Check if needed to include child runs:
        if include_children:
            outputs["child_runs"] = []
            for child_run in serialized_run.get("child_runs", []):
                # Recursively summarize the child run:
                summarized_child_run = self._summarize_run(serialized_run=child_run, include_children=True)
                if summarized_child_run:
                    inputs_child, outputs_child = summarized_child_run
                    outputs["child_runs"].append(
                        {
                            "input_data": inputs_child,
                            "output_data": outputs_child,
                        }
                    )

        return inputs, outputs

    def _send_run_event(
        self, event_id: str, inputs: dict, outputs: dict, start_time: datetime.datetime, end_time: datetime.datetime
    ):
        """
        Send a monitoring event for a single run.

        Note: If monitor debug mode is enabled, appends to ``debug_target_list`` or prints JSON.

        :param event_id: Unique event identifier.
        :param inputs: Inputs dictionary for the event.
        :param outputs: Outputs dictionary for the event.
        :param start_time: Request/start timestamp.
        :param end_time: Response/end timestamp.
        """
        event = {
            "event_id": event_id,
            "label": self._monitor_settings.label,
            "input_data": {"input_data": inputs},  # So it will be a single "input feature" in MLRun monitoring.
            "output_data": {"output_data": outputs},  # So it will be a single "output feature" in MLRun monitoring.
            "request_timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "response_timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S%z"),
        }
        if self._monitor_settings.debug:
            if isinstance(self._monitor_settings.debug_target_list, list):
                self._monitor_settings.debug_target_list.append(event)
            else:
                print(orjson.dumps(event, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
            return

        self._mlrun_client.monitor(**event)

    @staticmethod
    def _check_for_env_var_usage() -> bool:
        """
        Check whether global env-var activated tracing is requested.

        :returns: True when ``MLRUN_MONITORING_ENABLED`` environment variable equals ``"1"``.
        """
        return os.environ.get(mlrun_monitoring_env_var, "0") == "1"

    @staticmethod
    def _import_from_module_path(module_path: str) -> Any:
        """
        Import an object from a full module path string.

        :param module_path: Full dotted path, e.g. ``a.b.module.object``.

        :returns: The imported object.

        raise: ValueError: If ``module_path`` is not a valid Python module path.
        raise: ImportError: If module cannot be imported.
        raise: AttributeError: If the object name is not found in the module.
        """
        try:
            module_name, object_name = module_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            obj = getattr(module, object_name)
        except ValueError as value_error:
            raise ValueError(
                f"The provided '{module_path}' is not valid: it must have at least one '.'. "
                f"If the class is locally defined, please add '__main__.MyObject' to the path."
            ) from value_error
        except ImportError as import_error:
            raise ImportError(
                f"Could not import '{module_path}'. Tried to import '{module_name}' and failed with the following "
                f"error: {import_error}."
            ) from import_error
        except AttributeError as attribute_error:
            raise AttributeError(
                f"Could not import '{object_name}'. Tried to run 'from {module_name} import {object_name}' and could "
                f"not find it: {attribute_error}"
            ) from attribute_error

        return obj


#: MLRun monitoring context variable to set when the user wraps his code with `mlrun_monitoring`. From this context
# variable LangChain will get the tracer in a thread-safe way.
mlrun_monitoring_var: ContextVar[MLRunTracer | None] = ContextVar(
    "mlrun_monitoring", default=None
)


@contextmanager
def mlrun_monitoring(settings: MLRunTracerSettings | None = None):
    """
    Context manager to enable MLRun tracing for LangChain code to monitor LangChain runs.

    Example usage::

        from mlrun_tracer import mlrun_monitoring, MLRunTracerSettings

        settings = MLRunTracerSettings(...)
        with mlrun_monitoring(settings=settings) as tracer:
            # LangChain execution within this block will be traced by `tracer`.
            ...

    :param settings: The settings to use to configure the tracer.
    """
    mlrun_tracer = MLRunTracer(settings=settings)
    token = mlrun_monitoring_var.set(mlrun_tracer)
    try:
        yield mlrun_tracer
    finally:
        mlrun_monitoring_var.reset(token)


# Register a hook for LangChain to apply the MLRun tracer:
register_configure_hook(
    context_var=mlrun_monitoring_var,
    inheritable=True,  # To allow inner runs (agent that uses a tool that uses a llm...) to be traced.
    env_var=mlrun_monitoring_env_var,
    handle_class=MLRunTracer,
)


# Temporary convenient function to set up the monitoring infrastructure required for the tracer.
def setup_langchain_monitoring(
    project: str | mlrun.MlrunProject = None,
    function_name: str = "langchain_mlrun_function",
    model_name: str = "langchain_mlrun_model",
    model_endpoint_name: str = "langchain_mlrun_endpoint",
    monitoring_container: str = "projects",
    monitoring_stream_path: str = None,
) -> dict:
    """
    Create a model endpoint in the given project to be used for LangChain monitoring with MLRun and returns the
    necessary environment variables to configure the MLRun tracer client. The project should already exist and have
    monitoring enabled::

        project.set_model_monitoring_credentials(
            stream_profile_name=...,
            tsdb_profile_name=...
        )

    This function creates and logs dummy model and function in the specified project in order to create the model
    endpoint for monitoring. It is a temporary workaround and will be added as a feature in a future MLRun version.

    :param project: The MLRun project name or object where to create the model endpoint. If None, the current active
        project will be used.
    :param function_name: The name of the serving function to create.
    :param model_name: The name of the model to create.
    :param model_endpoint_name: The name of the model endpoint to create.
    :param monitoring_container: The V3IO container where the monitoring stream is located.
    :param monitoring_stream_path: The V3IO stream path for monitoring. If None,
        ``<project.name>/model-endpoints/stream-v1`` will be used.

    :returns: A dictionary with the necessary environment variables to configure the MLRun tracer client.

    raise: MLRunInvalidArgumentError: If no project is provided and there is no current active project.
    """
    import io
    import time
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    import tempfile
    import pickle
    import json

    from mlrun.common.helpers import parse_versioned_object_uri
    from mlrun.features import Feature

    class ProgressStep:
        """
        A context manager to display progress of a code block with timing and optional output suppression.
        """

        def __init__(self, label: str, indent: int = 2, width: int = 40, clean: bool = True):
            """
            Initialize the ProgressStep context manager.

            :param label: The label to display for the progress step.
            :param indent: The number of spaces to indent the label.
            :param width: The width to pad the label for alignment.
            :param clean: Whether to suppress stdout and stderr during the block execution.
            """
            # Store parameters:
            self._label = label
            self._indent = indent
            self._width = width
            self._clean = clean

            # Internal state:
            self._start_time = None
            self._sink = io.StringIO()
            self._stdout_redirect = None
            self._stderr_redirect = None
            self._last_line_length = 0  # To track the line printed when terminals don't support '\033[K'.

            # Capture the stream currently in use (before and if clean is true and we redirect it):
            self._terminal = sys.stdout

        def __enter__(self):
            """
            Enter the context manager, starting the timer and printing the initial status.
            """
            # Start timer:
            self._start_time = time.perf_counter()

            # Print without newline (using \r to allow overwriting):
            self._write(icon=" ", status="Running", new_line=False)

            # Silence all internal noise:
            if self._clean:
                self._stdout_redirect = redirect_stdout(self._sink)
                self._stderr_redirect = redirect_stderr(self._sink)
                self._stdout_redirect.__enter__()
                self._stderr_redirect.__enter__()

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Exit the context manager, stopping the timer and printing the final status.

            :param exc_type: The exception type, if any.
            :param exc_val: The exception value, if any.
            :param exc_tb: The exception traceback, if any.
            """
            # Restore stdout/stderr:
            if self._clean:
                self._stdout_redirect.__exit__(exc_type, exc_val, exc_tb)
                self._stderr_redirect.__exit__(exc_type, exc_val, exc_tb)

            # Calculate elapsed time:
            elapsed = time.perf_counter() - self._start_time

            # Move cursor back to start of line ('\r') and overwrite ('\033[K' clears the line to the right):
            if exc_type is None:
                self._write(icon="✓", status=f"Done ({elapsed:.2f}s)", new_line=True)
            else:
                self._write(icon="✕", status="Failed", new_line=True)

        def update(self, status: str):
            """
            Update the status message displayed for the progress step.

            :param status: The new status message to display.
            """
            self._write(icon=" ", status=status, new_line=False)

        def _write(self, icon: str, status: str, new_line: bool):
            """
            Write the progress line to the terminal, handling line clearing for terminals that do not support it.

            :param icon: The icon to display (e.g., checkmark, cross, space).
            :param status: The status message to display.
            :param new_line: Whether to end the line with a newline character.
            """
            # Construct the basic line
            line = f"\r{' ' * self._indent}[{icon}] {self._label.ljust(self._width, '.')} {status}"

            # Calculate if we need to pad with spaces to clear the old, longer line:
            padding = max(0, self._last_line_length - len(line))

            # Add spaces to clear old text (add the ANSI clear for terminals that support it):
            line = f"{line}{' ' * padding}\033[K"

            # Add newline if needed:
            if new_line:
                line += "\n"

            # Write to terminal:
            self._terminal.write(line)
            self._terminal.flush()

            # Update the max length seen so far:
            self._last_line_length = len(line)

    print("Creating LangChain model endpoint\n")

    # Get the project:
    with ProgressStep("Loading Project"):
        if project is None:
            try:
                project = mlrun.get_current_project(silent=False)
            except mlrun.errors.MLRunInvalidArgumentError:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "There is no current active project. Either use `mlrun.get_or_create_project` prior to "
                    "creating the monitoring endpoint or pass a project name to load."
                )
        if isinstance(project, str):
            project = mlrun.load_project(name=project)

    # Create and log the dummy model:
    with ProgressStep(f"Creating Model") as progress_step:
        # Check if the model already exists:
        progress_step.update("Checking if model exists")
        try:
            dummy_model = project.get_artifact(key=model_name)
        except mlrun.MLRunNotFoundError:
            dummy_model = None
        # If not, create and log it:
        if not dummy_model:
            progress_step.update(f"Logging model '{model_name}'")
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a dummy model file:
                dummy_model_path = os.path.join(tmpdir, "for_langchain_mlrun_tracer.pkl")
                with open(dummy_model_path, "wb") as f:
                    pickle.dump({"dummy": "model"}, f)
                # Log the model:
                dummy_model = project.log_model(
                    key=model_name,
                    model_file=dummy_model_path,
                    inputs=[Feature(value_type="str", name="input")],
                    outputs=[Feature(value_type='str', name="output")]
                )

    # Create and set the dummy function:
    with ProgressStep("Creating Function") as progress_step:
        # Check if the function already exists:
        progress_step.update("Checking if function exists")
        try:
            dummy_function = project.get_function(key=function_name)
        except mlrun.MLRunNotFoundError:
            dummy_function = None
        # If not, create and save it:
        if not dummy_function:
            progress_step.update(f"Setting function '{function_name}'")
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a dummy function file:
                dummy_function_code = """
def handler(context, event):
    return "ok"
"""
                dummy_function_path = os.path.join(tmpdir, "dummy_function.py")
                with open(dummy_function_path, "w") as f:
                    f.write(dummy_function_code)
                # Set the function in the project:
                dummy_function = project.set_function(
                    func=dummy_function_path, name=function_name, image="mlrun/mlrun", kind="nuclio"
                )
                dummy_function.save()

    # Create the model endpoint:
    with ProgressStep("Creating Model Endpoint") as progress_step:
        # Get the MLRun DB:
        progress_step.update("Getting MLRun DB")
        db = mlrun.get_run_db()
        # Check if the model endpoint already exists:
        progress_step.update("Checking if endpoint exists")
        model_endpoint = project.list_model_endpoints(names=[model_endpoint_name]).endpoints
        if model_endpoint:
            model_endpoint = model_endpoint[0]
        else:
            progress_step.update("Creating model endpoint")
            model_endpoint = mlrun.common.schemas.ModelEndpoint(
                metadata=mlrun.common.schemas.ModelEndpointMetadata(
                    project=project.name,
                    name=model_endpoint_name,
                    endpoint_type=mlrun.common.schemas.model_monitoring.EndpointType.NODE_EP,
                ),
                spec=mlrun.common.schemas.ModelEndpointSpec(
                    function_name=dummy_function.metadata.name,
                    function_tag="latest",
                    model_path=dummy_model.uri,
                    model_class="CustomStream",
                ),
                status=mlrun.common.schemas.ModelEndpointStatus(
                    monitoring_mode=mm_constants.ModelMonitoringMode.enabled,
                ),
            )
            db.create_model_endpoint(model_endpoint=model_endpoint)
            # Wait for the model endpoint UID to be set:
            progress_step.update("Waiting for model endpoint")
            uid_exist_flag = False
            while not uid_exist_flag:
                model_endpoint = project.list_model_endpoints(names=[model_endpoint_name])
                model_endpoint = model_endpoint.endpoints[0]
                if model_endpoint.metadata.uid:
                    uid_exist_flag = True

    # Prepare the environment variables:
    monitoring_stream_path = monitoring_stream_path or f"{project.name}/model-endpoints/stream-v1"
    env_vars = {
        "MLRUN_MONITORING_ENABLED": "1",
        "MLRUN_TRACER_CLIENT_PROJECT": project.name,
        "MLRUN_TRACER_CLIENT_STREAM_PATH": monitoring_stream_path,
        "MLRUN_TRACER_CLIENT_CONTAINER": monitoring_container,
        "MLRUN_TRACER_CLIENT_MODEL_ENDPOINT_NAME": model_endpoint.metadata.name,
        "MLRUN_TRACER_CLIENT_MODEL_ENDPOINT_UID": model_endpoint.metadata.uid,
        "MLRUN_TRACER_CLIENT_SERVING_FUNCTION": function_name,
    }
    print("\n✨ Done! LangChain monitoring model endpoint created successfully.")
    print("You can now set the following environment variables to enable MLRun tracing in your LangChain code:\n")
    print(json.dumps(env_vars, indent=4))
    print(
        "\nTo customize the monitoring behavior, you can also set additional environment variables prefixed with "
        "'MLRUN_TRACER_MONITOR_'. Refer to the MLRun tracer documentation for more details.\n"
    )

    return env_vars


class LangChainMonitoringApp(ModelMonitoringApplicationBase):
    """
    A base monitoring application for LangChain that calculates common metrics on LangChain runs traced with the MLRun
    tracer.

    The class is inheritable and can be extended to add custom metrics or override existing ones. It provides methods to
    extract structured runs from the monitoring context and calculate metrics such as average latency, success rate,
    token usage, and run name counts.

    If inheriting, the main method to override is `do_tracking`, which performs the tracking on the monitoring context.
    """

    def do_tracking(self, monitoring_context: MonitoringApplicationContext) -> (
        ModelMonitoringApplicationResult |
        list[ModelMonitoringApplicationResult | ModelMonitoringApplicationMetric] |
        dict[str, Any]
    ):
        """
        The main function that performs tracking on the monitoring context. The LangChain monitoring app by default
        will calculate all the provided metrics on the structured runs extracted from the monitoring context sample
        dataframe.

        :param monitoring_context: The monitoring context containing the sample dataframe.

        :returns: The monitoring artifacts, metrics and results.
        """
        # Get the structured runs from the monitoring context:
        structured_runs, _ = self.get_structured_runs(monitoring_context=monitoring_context)

        # Calculate the metrics:
        average_latency = self.calculate_average_latency(structured_runs=structured_runs)
        success_rate = self.calculate_success_rate(structured_runs=structured_runs)
        token_usage = self.count_token_usage(structured_runs=structured_runs)
        run_name_counts = self.count_run_names(structured_runs=structured_runs)

        return [
            ModelMonitoringApplicationMetric(
                name="average_latency",
                value=average_latency,
            ),
            ModelMonitoringApplicationMetric(
                name="success_rate",
                value=success_rate,
            ),
            ModelMonitoringApplicationMetric(
                name="total_input_tokens",
                value=token_usage["total_input_tokens"],
            ),
            ModelMonitoringApplicationMetric(
                name="total_output_tokens",
                value=token_usage["total_output_tokens"],
            ),
            ModelMonitoringApplicationMetric(
                name="combined_total_tokens",
                value=token_usage["combined_total"],
            ),
            *[ModelMonitoringApplicationMetric(
                name=f"run_name_counts_{run_name}",
                value=count,
            ) for run_name, count in run_name_counts.items()],
        ]

    @staticmethod
    def get_structured_runs(
        monitoring_context: MonitoringApplicationContext,
        labels_filter: list[str] = None,
        tags_filter: list[str] = None,
        run_name_filter: list[str] = None,
        run_type_filter: list[str] = None,
        flatten_child_runs: bool = False,
        ignore_child_runs: bool = False,
        ignore_errored_runs: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """
        Get the structured runs from the monitoring context sample dataframe. The sample dataframe contains the raw
        input and output data as JSON strings - the way the MLRun tracer sends them as events to MLRun monitoring. This
        function parses the JSON strings into structured dictionaries that can be used for further metrics calculations
        and analysis.

        :param monitoring_context: The monitoring context containing the sample dataframe.
        :param labels_filter: List of labels to filter the runs. Only runs with a label appearing in this list will
            remain. If None, no filtering is applied.
        :param tags_filter: List of tags to filter the runs. Only runs containing at least one tag from this list will
            remain. If None, no filtering is applied.
        :param run_name_filter: List of run names to filter the runs. Only runs with a name appearing in this list will
            remain. If None, no filtering is applied.
        :param run_type_filter: List of run types to filter the runs. Only runs with a type appearing in this list will
            remain. If None, no filtering is applied.
        :param flatten_child_runs: Whether to flatten child runs into the main runs list. If True, all child runs will
            be extracted and added to the main runs list. If False, child runs will be kept nested within their parent
            runs.
        :param ignore_child_runs: Whether to ignore child runs completely. If True, child runs will be removed from the
            output. If False, child runs will be processed according to the other parameters.
        :param ignore_errored_runs: Whether to ignore runs that resulted in errors. If True, runs with errors will be
            excluded from the output. If False, errored runs will be included.

        :returns: A list of structured run dictionaries that passed the filters and a list of samples that could not be
            parsed due to errors.
        """
        # Retrieve the input and output samples from the monitoring context:
        samples = monitoring_context.sample_df[['input', 'output']].to_dict('records')

        # Prepare to collect structured samples:
        structured_samples = []
        errored_samples = []

        # Go over all samples:
        for sample in samples:
            try:
                # Parse the input data into structured format:
                parsed_input = orjson.loads(sample['input'])
                label = parsed_input['label']
                parsed_input = parsed_input["input"]["input_data"]
                # Parse the output data into structured format:
                parsed_output = orjson.loads(sample['output'])["output_data"]
                structured_samples.extend(
                    LangChainMonitoringApp._collect_run(
                        structured_input=parsed_input,
                        structured_output=parsed_output,
                        label=label,
                        labels_filter=labels_filter,
                        tags_filter=tags_filter,
                        run_name_filter=run_name_filter,
                        run_type_filter=run_type_filter,
                        flatten_child_runs=flatten_child_runs,
                        ignore_child_runs=ignore_child_runs,
                        ignore_errored_runs=ignore_errored_runs,
                    )
                )
            except Exception:
                errored_samples.append(sample)

        return structured_samples, errored_samples

    @staticmethod
    def _collect_run(
        structured_input: dict,
        structured_output: dict,
        label: str,
        child_level: int = 0,
        labels_filter: list[str] = None,
        tags_filter: list[str] = None,
        run_name_filter: list[str] = None,
        run_type_filter: list[str] = None,
        flatten_child_runs: bool = False,
        ignore_child_runs: bool = False,
        ignore_errored_runs: bool = False,
    ) -> list[dict]:
        """
        Recursively collect runs from the structured input and output data, applying filters as specified.

        :param structured_input: The structured input data of the run.
        :param structured_output: The structured output data of the run.
        :param label: The label of the run.
        :param child_level: The current child level of the run (0 for root runs).
        :param labels_filter: Label filter as described in `get_structured_runs`.
        :param tags_filter: Tag filter as described in `get_structured_runs`.
        :param run_name_filter: Run name filter as described in `get_structured_runs`.
        :param run_type_filter: Run type filter as described in `get_structured_runs`.
        :param flatten_child_runs: Flag to flatten child runs as described in `get_structured_runs`.
        :param ignore_child_runs: Flag to ignore child runs as described in `get_structured_runs`.
        :param ignore_errored_runs: Flag to ignore errored runs as described in `get_structured_runs`.

        :returns: A list of structured run dictionaries that passed the filters.
        """
        # Prepare to collect runs:
        runs = []

        # Filter by label:
        if labels_filter and label not in labels_filter:
            return runs

        # Handle child runs:
        if "child_runs" in structured_output:
            # Check if we need to ignore or flatten child runs:
            if ignore_child_runs:
                structured_output.pop("child_runs")
            elif flatten_child_runs:
                # Recursively collect child runs:
                child_runs = structured_output.pop("child_runs")
                flattened_runs = []
                for child_run in child_runs:
                    flattened_runs.extend(
                        LangChainMonitoringApp._collect_run(
                            structured_input=child_run["input_data"],
                            structured_output=child_run["output_data"],
                            label=label,
                            child_level=child_level + 1,
                            tags_filter=tags_filter,
                            run_name_filter=run_name_filter,
                            run_type_filter=run_type_filter,
                            flatten_child_runs=flatten_child_runs,
                            ignore_child_runs=ignore_child_runs,
                            ignore_errored_runs=ignore_errored_runs,
                        )
                    )
                runs.extend(flattened_runs)

        # Filter by tags, run name, run type, and errors:
        if tags_filter and not set(structured_input["tags"]).isdisjoint(tags_filter):
            return runs
        if run_name_filter and structured_input["run_name"] not in run_name_filter:
            return runs
        if run_type_filter and structured_input["run_type"] not in run_type_filter:
            return runs
        if ignore_errored_runs and structured_output.get("error", None):
            return runs

        # Collect the current run:
        runs.append({"label": label, "input_data": structured_input, "output_data": structured_output,
                     "child_level": child_level})
        return runs

    @staticmethod
    def iterate_structured_runs(structured_runs: list[dict]) -> Generator[dict, None, None]:
        """
        Iterates over all runs in the structured samples, including child runs.

        :param structured_runs: List of structured run samples.

        :returns: A generator yielding each run structure.
        """
        # TODO: Add an option to stop at a certain child level.
        for structured_run in structured_runs:
            if "child_runs" in structured_run['output_data']:
                for child_run in structured_run['output_data']['child_runs']:
                    yield from LangChainMonitoringApp.iterate_structured_runs([{
                        "label": structured_run['label'],
                        "input_data": child_run['input_data'],
                        "output_data": child_run['output_data'],
                        "child_level": structured_run['child_level'] + 1
                    }])
            yield structured_run

    @staticmethod
    def count_run_names(structured_runs: list[dict]) -> dict[str, int]:
        """
        Counts occurrences of each run name in the structured samples.

        :param structured_runs: List of structured run samples.

        :returns: A dictionary with run names as keys and their counts as values.
        """
        # TODO: Add a nice plot artifact that will draw the bar chart for what is being used the most.
        # Prepare to count run names:
        run_name_counts = {}

        # Go over all the runs:
        for structured_run in LangChainMonitoringApp.iterate_structured_runs(structured_runs):
            run_name = structured_run['input_data']['run_name']
            if run_name in run_name_counts:
                run_name_counts[run_name] += 1
            else:
                run_name_counts[run_name] = 1

        return run_name_counts

    @staticmethod
    def count_token_usage(structured_runs: list[dict]) -> dict:
        """
        Calculates total tokens by only counting unique 'llm' type runs.

        :param structured_runs: List of structured run samples.

        :returns: A dictionary with total input tokens, total output tokens, and combined total tokens.
        """
        # TODO: Add a token count per model breakdown (a dictionary of <model_provider>:<model_name> to token counts)
        #       including an artifact that will plot it nicely. Pay attention that different providers use different
        #       keys in the response metadata. We should implement a mapping for that so each provider will have its own
        #       handler that will know how to extract the relevant info out of a run.
        # Prepare to count tokens:
        total_input_tokens = 0
        total_output_tokens = 0

        # Go over all the LLM typed runs:
        for structured_run in LangChainMonitoringApp.iterate_structured_runs(structured_runs):
            # Count only LLM type runs as chain runs may include duplicative information as they accumulate the tokens
            # from the child runs:
            if structured_run['input_data']['run_type'] != 'llm':
                continue
            # Look for the token count information:
            outputs = structured_run['output_data']["outputs"]
            # Newer implementations should have the metadata in the `AIMessage` kwargs under generations:
            if "generations" in outputs:
                for generation in outputs["generations"]:  # Iterate over generations.
                    for sample in generation:  # Iterate over the generation batch.
                        token_usage = sample.get("message", {}).get("kwargs", {}).get("usage_metadata", {})
                        if token_usage:
                            total_input_tokens += (
                                token_usage.get('input_tokens', 0)
                                or token_usage.get('prompt_tokens', 0)
                            )
                            total_output_tokens += (
                                    token_usage.get('output_tokens', 0) or
                                    token_usage.get('completion_tokens', 0)
                            )
                continue
            # Older implementations may have the metadata under `llm_output`:
            if "llm_output" in outputs:
                token_usage = outputs["llm_output"].get("token_usage", {})
                if token_usage:
                    total_input_tokens += token_usage.get('input_tokens', 0) or token_usage.get('prompt_tokens', 0)
                    total_output_tokens += (
                        token_usage.get('output_tokens', 0) or
                        token_usage.get('completion_tokens', 0)
                    )

        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "combined_total": total_input_tokens + total_output_tokens
        }

    @staticmethod
    def calculate_success_rate(structured_runs: list[dict]) -> float:
        """
        Calculates the success rate across all runs.

        :param structured_runs: List of structured run samples.

        :returns: Success rate as a float percentage between 0 and 1.
        """
        # TODO: Add an option to see errors breakdown by kind of error and maybe an option to show which run name yielded
        #       most of the errors with artifacts showcasing it.
        successful_count = 0
        for structured_run in structured_runs:
            if 'error' not in structured_run['output_data'] or structured_run['output_data']['error'] is None:
                successful_count += 1
        return successful_count / len(structured_runs) if structured_runs else 0.0

    @staticmethod
    def calculate_average_latency(structured_runs: list[dict]) -> float:
        """
        Calculates the average latency across all runs.

        :param structured_runs: List of structured run samples.

        :returns: Average latency in milliseconds.
        """
        # TODO: Add an option to calculate latency per run name (to know which runs are slower/faster) and then return an
        #       artifact showcasing it.
        # Prepare to calculate average latency:
        total_latency = 0.0
        count = 0

        # Go over all the root runs:
        for structured_run in structured_runs:
            # Skip child runs:
            if structured_run["child_level"] > 0:
                continue
            # Check if latency is already provided:
            if "latency" in structured_run['output_data']:
                total_latency += structured_run['output_data']['latency']
                count += 1
                continue
            # Calculate latency from timestamps:
            start_time = datetime.datetime.fromisoformat(structured_run['input_data']['start_timestamp'])
            end_time = datetime.datetime.fromisoformat(structured_run['output_data']['end_timestamp'])
            total_latency += (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds
            count += 1

        return total_latency / count if count > 0 else 0.0
