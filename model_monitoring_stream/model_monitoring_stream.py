import json
from collections import defaultdict
from datetime import datetime
from os import environ
from typing import Dict, List, Set, Optional

import pandas as pd
from mlrun.config import config
from mlrun.run import MLClientCtx
from mlrun.utils import logger
from mlrun.utils.model_monitoring import (
    parse_model_endpoint_store_prefix,
    create_model_endpoint_id,
)
from mlrun.utils.v3io_clients import get_v3io_client, get_frames_client
from nuclio import Event
from storey import (
    FieldAggregator,
    NoopDriver,
    Table,
    Source,
    Map,
    MapClass,
    AggregateByKey,
    build_flow,
    FlatMap,
    WriteToParquet,
    Filter,
    WriteToTSDB,
)
from storey.dtypes import SlidingWindows
from storey.steps import SampleWindow

# Constants
ISO_8061_UTC = "%Y-%m-%d %H:%M:%S.%f%z"
FUNCTION_URI = "function_uri"
MODEL = "model"
VERSION = "version"
VERSIONED_MODEL = "versioned_model"
MODEL_CLASS = "model_class"
TIMESTAMP = "timestamp"
ENDPOINT_ID = "endpoint_id"
REQUEST_ID = "request_id"
LABELS = "labels"
UNPACKED_LABELS = "unpacked_labels"
LATENCY_AVG_5M = "latency_avg_5m"
LATENCY_AVG_1H = "latency_avg_1h"
PREDICTIONS_PER_SECOND = "predictions_per_second"
PREDICTIONS_COUNT_5M = "predictions_count_5m"
PREDICTIONS_COUNT_1H = "predictions_count_1h"
FIRST_REQUEST = "first_request"
LAST_REQUEST = "last_request"
ERROR_COUNT = "error_count"
ENTITIES = "entities"
FEATURE_NAMES = "feature_names"
LATENCY = "latency"
RECORD_TYPE = "record_type"
FEATURES = "features"
PREDICTION = "prediction"
PREDICTIONS = "predictions"
NAMED_FEATURES = "named_features"
BASE_METRICS = "base_metrics"
CUSTOM_METRICS = "custom_metrics"
ENDPOINT_FEATURES = "endpoint_features"
METRICS = "metrics"
BATCH_TIMESTAMP = "batch_timestamp"
TIME_FORMAT: str = "%Y-%m-%d %H:%M:%S.%f"  # ISO 8061


# Stream processing code
class EventStreamProcessor:
    def __init__(
        self,
        project: str,
        sample_window: int = 10,
        tsdb_batching_max_events: int = 10,
        tsdb_batching_timeout_secs: int = 60 * 5,  # Default 5 minutes
        parquet_batching_max_events: int = 10_000,
        parquet_batching_timeout_secs: int = 60 * 60,  # Default 1 hour
        aggregate_count_windows: Optional[List[str]] = None,
        aggregate_count_period: str = "30s",
        aggregate_avg_windows: Optional[List[str]] = None,
        aggregate_avg_period: str = "30s",
        v3io_access_key: Optional[str] = None,
        v3io_framesd: Optional[str] = None,
    ):
        self.project = project
        self.sample_window = sample_window
        self.tsdb_batching_max_events = tsdb_batching_max_events
        self.tsdb_batching_timeout_secs = tsdb_batching_timeout_secs
        self.parquet_batching_max_events = parquet_batching_max_events
        self.parquet_batching_timeout_secs = parquet_batching_timeout_secs
        self.aggregate_count_windows = aggregate_count_windows or ["5m", "1h"]
        self.aggregate_count_period = aggregate_count_period
        self.aggregate_avg_windows = aggregate_avg_windows or ["5m", "1h"]
        self.aggregate_avg_period = aggregate_avg_period
        self.v3io_access_key = v3io_access_key or environ.get("V3IO_ACCESS_KEY")
        self.v3io_framesd = v3io_framesd or environ.get("V3IO_FRAMESD")

        template = config.model_endpoint_monitoring.store_prefixes.default

        kv_path = template.format(project=project, kind="endpoints")
        _, self.kv_container, self.kv_path = parse_model_endpoint_store_prefix(kv_path)

        tsdb_path = template.format(project=project, kind="events")
        _, self.tsdb_container, self.tsdb_path = parse_model_endpoint_store_prefix(
            tsdb_path
        )
        self.tsdb_path = f"{self.tsdb_container}/{self.tsdb_path}"

        self.parquet_path = template.format(project=project, kind="parquet")

        logger.info(
            "Writer paths",
            kv_path=self.kv_path,
            tsdb_path=self.tsdb_path,
            parquet_path=self.parquet_path,
        )

        self._kv_keys = [
            FUNCTION_URI,
            MODEL,
            MODEL_CLASS,
            TIMESTAMP,
            ENDPOINT_ID,
            LABELS,
            UNPACKED_LABELS,
            LATENCY_AVG_5M,
            LATENCY_AVG_1H,
            PREDICTIONS_PER_SECOND,
            PREDICTIONS_COUNT_5M,
            PREDICTIONS_COUNT_1H,
            FIRST_REQUEST,
            LAST_REQUEST,
            ERROR_COUNT,
        ]

        self._flow = build_flow(
            [
                Source(),
                ProcessEndpointEvent(self.kv_container, self.kv_path),
                FilterNotNone(),
                FlattenPredictions(),
                MapFeatureNames(self.kv_container, self.kv_path),
                # Branch 1: Aggregate events, count averages and update TSDB and KV
                [
                    AggregateByKey(
                        aggregates=[
                            FieldAggregator(
                                PREDICTIONS,
                                ENDPOINT_ID,
                                ["count"],
                                SlidingWindows(
                                    self.aggregate_count_windows,
                                    self.aggregate_count_period,
                                ),
                            ),
                            FieldAggregator(
                                LATENCY,
                                LATENCY,
                                ["avg"],
                                SlidingWindows(
                                    self.aggregate_avg_windows,
                                    self.aggregate_avg_period,
                                ),
                            ),
                        ],
                        table=Table("notable", NoopDriver()),
                    ),
                    SampleWindow(
                        self.sample_window
                    ),  # Add required gap between event to apply sampling
                    Map(self.compute_predictions_per_second),
                    # Branch 1.1: Updated KV
                    [
                        Map(self.process_before_kv),
                        WriteToKV(container=self.kv_container, table=self.kv_path),
                        InferSchema(
                            v3io_access_key=self.v3io_access_key,
                            v3io_framesd=self.v3io_framesd,
                            container=self.kv_container,
                            table=self.kv_path,
                        ),
                    ],
                    # Branch 1.2: Update TSDB
                    [
                        # Map the event into taggable fields, add record type to each field
                        Map(self.process_before_events_tsdb),
                        [
                            FilterKeys(BASE_METRICS),
                            UnpackValues(BASE_METRICS),
                            WriteToTSDB(
                                path=self.tsdb_path,
                                rate="10/m",
                                time_col=TIMESTAMP,
                                container=self.tsdb_container,
                                access_key=self.v3io_access_key,
                                v3io_frames=self.v3io_framesd,
                                index_cols=[ENDPOINT_ID, RECORD_TYPE],
                                # Settings for _Batching
                                max_events=self.tsdb_batching_max_events,
                                timeout_secs=self.tsdb_batching_timeout_secs,
                                key=ENDPOINT_ID,
                            ),
                        ],
                        [
                            FilterKeys(ENDPOINT_FEATURES),
                            UnpackValues(ENDPOINT_FEATURES),
                            WriteToTSDB(
                                path=self.tsdb_path,
                                rate="10/m",
                                time_col=TIMESTAMP,
                                container=self.tsdb_container,
                                access_key=self.v3io_access_key,
                                v3io_frames=self.v3io_framesd,
                                index_cols=[ENDPOINT_ID, RECORD_TYPE],
                                # Settings for _Batching
                                max_events=self.tsdb_batching_max_events,
                                timeout_secs=self.tsdb_batching_timeout_secs,
                                key=ENDPOINT_ID,
                            ),
                        ],
                        [
                            FilterKeys(CUSTOM_METRICS),
                            FilterNotNone(),
                            UnpackValues(CUSTOM_METRICS),
                            WriteToTSDB(
                                path=self.tsdb_path,
                                rate="10/m",
                                time_col=TIMESTAMP,
                                container=self.tsdb_container,
                                access_key=self.v3io_access_key,
                                v3io_frames=self.v3io_framesd,
                                index_cols=[ENDPOINT_ID, RECORD_TYPE],
                                # Settings for _Batching
                                max_events=self.tsdb_batching_max_events,
                                timeout_secs=self.tsdb_batching_timeout_secs,
                                key=ENDPOINT_ID,
                            ),
                        ],
                    ],
                ],
                # Branch 2: Batch events, write to parquet
                [
                    Map(self.process_before_parquet),
                    WriteToParquet(
                        path=self.parquet_path,
                        infer_columns_from_data=True,
                        # Settings for _Batching
                        max_events=self.parquet_batching_max_events,
                        timeout_secs=self.parquet_batching_timeout_secs,
                    ),
                ],
            ]
        ).run()

    def consume(self, event: Dict):
        events = []
        if "headers" in event and "values" in event:
            for values in event["values"]:
                events.append({k: v for k, v in zip(event["headers"], values)})
        else:
            events.append(event)

        for enriched in map(enrich_even_details, events):
            if enriched is not None:
                self._flow.emit(
                    enriched,
                    key=enriched[ENDPOINT_ID],
                    event_time=datetime.strptime(enriched["when"], ISO_8061_UTC),
                )
            else:
                pass

    @staticmethod
    def compute_predictions_per_second(event: dict):
        event[PREDICTIONS_PER_SECOND] = float(event[PREDICTIONS_COUNT_5M]) / 600
        return event

    def process_before_kv(self, event: dict):
        # Filter relevant keys
        e = {k: event[k] for k in self._kv_keys}
        # Unpack labels dictionary
        e = {**e, **e.pop(UNPACKED_LABELS, {})}
        # Write labels to kv as json string to be presentable later
        e[LABELS] = json.dumps(e[LABELS])
        return e

    @staticmethod
    def process_before_events_tsdb(event: Dict):
        base_fields = [TIMESTAMP, ENDPOINT_ID]

        base_event = {k: event[k] for k in base_fields}
        base_event[TIMESTAMP] = pd.to_datetime(
            base_event[TIMESTAMP], format=TIME_FORMAT
        )

        base_metrics = {
            RECORD_TYPE: BASE_METRICS,
            PREDICTIONS_PER_SECOND: event[PREDICTIONS_PER_SECOND],
            PREDICTIONS_COUNT_5M: event[PREDICTIONS_COUNT_5M],
            PREDICTIONS_COUNT_1H: event[PREDICTIONS_COUNT_1H],
            LATENCY_AVG_5M: event[LATENCY_AVG_5M],
            LATENCY_AVG_1H: event[LATENCY_AVG_1H],
            **base_event,
        }

        endpoint_features = {
            RECORD_TYPE: ENDPOINT_FEATURES,
            PREDICTION: event[PREDICTION],
            **event[NAMED_FEATURES],
            **base_event,
        }

        processed = {BASE_METRICS: base_metrics, ENDPOINT_FEATURES: endpoint_features}

        if event[METRICS]:
            processed[CUSTOM_METRICS] = {
                RECORD_TYPE: CUSTOM_METRICS,
                **event[METRICS],
                **base_event,
            }

        return processed

    @staticmethod
    def process_before_parquet(event: dict):
        def set_none_if_empty(_event: dict, keys: List[str]):
            for key in keys:
                if not _event.get(key):
                    _event[key] = None

        def drop_if_exists(_event: dict, keys: List[str]):
            for key in keys:
                _event.pop(key, None)

        def unpack_if_exists(_event: dict, keys: List[str]):
            for key in keys:
                value = _event.get(key)
                if value is not None:
                    _event = {**value, **event}

        drop_if_exists(event, [UNPACKED_LABELS, FEATURES])
        unpack_if_exists(event, [ENTITIES])
        set_none_if_empty(event, [LABELS, METRICS, ENTITIES])
        return event


class ProcessEndpointEvent(MapClass):
    def __init__(self, kv_container: str, kv_path: str, **kwargs):
        super().__init__(**kwargs)
        self.kv_container: str = kv_container
        self.kv_path: str = kv_path
        self.first_request: Dict[str, str] = dict()
        self.last_request: Dict[str, str] = dict()
        self.error_count: Dict[str, int] = defaultdict(int)
        self.endpoints: Set[str] = set()

    def do(self, event: dict):
        function_uri = event[FUNCTION_URI]
        versioned_model = event[VERSIONED_MODEL]
        endpoint_id = event[ENDPOINT_ID]

        # In case this process fails, resume state from existing record
        self.resume_state(endpoint_id)

        # Handle errors coming from stream
        found_errors = self.handle_errors(endpoint_id, event)
        if found_errors:
            return None

        # Validate event fields
        model_class = event.get("model_class") or event.get("class")
        timestamp = event.get("when")
        request_id = event.get("request", {}).get("id")
        latency = event.get("microsec")
        features = event.get("request", {}).get("inputs")
        prediction = event.get("resp", {}).get("outputs")

        if not self.is_valid_or_count(timestamp, ["when"]):
            return None

        if endpoint_id not in self.first_request:
            self.first_request[endpoint_id] = timestamp
        self.last_request[endpoint_id] = timestamp

        if not self.is_valid_or_count(request_id, ["request", "id"]):
            return None
        if not self.is_valid_or_count(latency, ["microsec"]):
            return None
        if not self.is_valid_or_count(features, ["request", "inputs"]):
            return None
        if not self.is_valid_or_count(prediction, ["resp", "outputs"]):
            return None

        event = {
            FUNCTION_URI: function_uri,
            MODEL: versioned_model,
            MODEL_CLASS: model_class,
            TIMESTAMP: timestamp,
            ENDPOINT_ID: endpoint_id,
            REQUEST_ID: request_id,
            LATENCY: latency,
            FEATURES: features,
            PREDICTION: prediction,
            FIRST_REQUEST: self.first_request[endpoint_id],
            LAST_REQUEST: self.last_request[endpoint_id],
            ERROR_COUNT: self.error_count[endpoint_id],
            LABELS: event.get(LABELS, {}),
            METRICS: event.get(METRICS, {}),
            ENTITIES: event.get("request", {}).get(ENTITIES, {}),
            UNPACKED_LABELS: {f"_{k}": v for k, v in event.get(LABELS, {}).items()},
        }

        return event

    def resume_state(self, endpoint_id):
        # Make sure process is resumable, if process fails for any reason, be able to pick things up close to where we
        # left them
        if endpoint_id not in self.endpoints:
            logger.info("Trying to resume state", endpoint_id=endpoint_id)
            endpoint_record = get_endpoint_record(
                kv_container=self.kv_container,
                kv_path=self.kv_path,
                endpoint_id=endpoint_id,
            )
            if endpoint_record:
                first_request = endpoint_record.get(FIRST_REQUEST)
                if first_request:
                    self.first_request[endpoint_id] = first_request
                error_count = endpoint_record.get(ERROR_COUNT)
                if error_count:
                    self.error_count[endpoint_id] = error_count
            self.endpoints.add(endpoint_id)

    def is_valid_or_count(self, field: str, dict_path: List[str]):
        if not is_valid_input(field, dict_path):
            self.error_count += 1
            return False
        return True

    def handle_errors(self, endpoint_id, event) -> bool:
        if "error" in event:
            self.error_count += 1
            return True

        return False


def enrich_even_details(event) -> Optional[dict]:
    function_uri = event.get(FUNCTION_URI)

    if not is_valid_input(function_uri, [FUNCTION_URI]):
        return None

    model = event.get(MODEL)
    if not is_valid_input(model, [MODEL]):
        return None

    version = event.get(VERSION)
    versioned_model = f"{model}:{version}" if version else model

    endpoint_id = create_model_endpoint_id(
        function_uri=function_uri, versioned_model=versioned_model,
    )

    endpoint_id = str(endpoint_id)

    event[VERSIONED_MODEL] = versioned_model
    event[ENDPOINT_ID] = endpoint_id

    return event


def is_valid_input(field, dict_path: List[str]):
    if field is None:
        logger.error(
            f"Expected event field is missing: {field} [Event -> {''.join(dict_path)}]"
        )
        return False
    return True


class FlattenPredictions(FlatMap):
    def __init__(self, **kwargs):
        super().__init__(fn=FlattenPredictions.flatten, **kwargs)

    @staticmethod
    def flatten(event: Dict):
        predictions = []
        for features, prediction in zip(event[FEATURES], event[PREDICTION]):
            predictions.append(dict(event, features=features, prediction=prediction))
        return predictions


class FilterNotNone(Filter):
    def __init__(self, **kwargs):
        super().__init__(fn=lambda event: event is not None, **kwargs)


class FilterKeys(MapClass):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.keys = list(args)

    def do(self, event):
        new_event = {}
        for key in self.keys:
            if key in event:
                new_event[key] = event[key]

        return new_event if new_event else None


class UnpackValues(MapClass):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.keys_to_unpack = set(args)

    def do(self, event):
        unpacked = {}
        for key in event.keys():
            if key in self.keys_to_unpack:
                unpacked = {**unpacked, **event[key]}
            else:
                unpacked[key] = event[key]
        return unpacked


class MapFeatureNames(MapClass):
    def __init__(self, kv_container: str, kv_path: str, **kwargs):
        super().__init__(**kwargs)
        self.kv_container = kv_container
        self.kv_path = kv_path
        self.feature_names = {}

    def do(self, event: Dict):
        endpoint_id = event[ENDPOINT_ID]

        if endpoint_id not in self.feature_names:
            endpoint_record = get_endpoint_record(
                kv_container=self.kv_container,
                kv_path=self.kv_path,
                endpoint_id=endpoint_id,
            )
            feature_names = endpoint_record.get(FEATURE_NAMES)
            feature_names = json.loads(feature_names) if feature_names else None

            if not feature_names:
                logger.warn(
                    f"Seems like endpoint {event[ENDPOINT_ID]} was not registered, feature names will be "
                    f"automatically generated"
                )
                feature_names = [f"f{i}" for i, _ in enumerate(event[FEATURES])]
                get_v3io_client().kv.update(
                    container=self.kv_container,
                    table_path=self.kv_path,
                    key=event[ENDPOINT_ID],
                    attributes={FEATURE_NAMES: json.dumps(feature_names)},
                )

            self.feature_names[endpoint_id] = feature_names

        feature_names = self.feature_names[endpoint_id]
        features = event[FEATURES]
        event[NAMED_FEATURES] = {
            name: feature for name, feature in zip(feature_names, features)
        }
        return event


class WriteToKV(MapClass):
    def __init__(self, container: str, table: str, **kwargs):
        super().__init__(**kwargs)
        self.container = container
        self.table = table

    def do(self, event: Dict):
        get_v3io_client().kv.update(
            container=self.container,
            table_path=self.table,
            key=event[ENDPOINT_ID],
            attributes=event,
        )
        return event


class InferSchema(MapClass):
    def __init__(
        self,
        v3io_access_key: str,
        v3io_framesd: str,
        container: str,
        table: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.container = container
        self.v3io_access_key = v3io_access_key
        self.v3io_framesd = v3io_framesd
        self.table = table
        self.keys = set()

    def do(self, event: Dict):
        key_set = set(event.keys())
        if not key_set.issubset(self.keys):
            self.keys.update(key_set)
            get_frames_client(
                token=self.v3io_access_key,
                container=self.container,
                address=self.v3io_framesd,
            ).execute(backend="kv", table=self.table, command="infer_schema")
            logger.info(
                "Found new keys, inferred schema", table=self.table, event=event
            )
        return event


def get_endpoint_record(
    kv_container: str, kv_path: str, endpoint_id: str
) -> Optional[dict]:
    logger.info(
        f"Grabbing endpoint data", endpoint_id=endpoint_id, table_path=kv_path,
    )
    try:
        endpoint_record = (
            get_v3io_client()
            .kv.get(container=kv_container, table_path=kv_path, key=endpoint_id,)
            .output.item
        )
        return endpoint_record
    except Exception:
        return None


def init_context(context: MLClientCtx):
    context.logger.info("Initializing EventStreamProcessor")
    parameters = environ.get("MODEL_MONITORING_PARAMETERS")
    parameters = json.loads(parameters) if parameters else {}
    stream_processor = EventStreamProcessor(**parameters)
    setattr(context, "stream_processor", stream_processor)


def handler(context: MLClientCtx, event: Event):
    event_body = json.loads(event.body)
    context.logger.info(event_body)
    context.stream_processor.consume(event_body)
