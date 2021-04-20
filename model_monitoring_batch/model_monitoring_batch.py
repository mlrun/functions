import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from mlrun import get_run_db
from mlrun import store_manager
from mlrun.data_types.infer import DFDataInfer, InferOptions
from mlrun.run import MLClientCtx
from mlrun.utils import logger, config
from mlrun.utils.model_monitoring import parse_model_endpoint_store_prefix
from mlrun.utils.v3io_clients import get_v3io_client
from sklearn.preprocessing import KBinsDiscretizer

ISO_8061 = "%Y-%m-%d %H:%M:%S.%f"


@dataclass
class TotalVarianceDistance:
    """
    Provides a symmetric drift distance between two periods t and u
    Z - vector of random variables
    Pt - Probability distribution over time span t
    """

    distrib_t: np.ndarray
    distrib_u: np.ndarray

    def compute(self) -> float:
        return np.sum(np.abs(self.distrib_t - self.distrib_u)) / 2


@dataclass
class HellingerDistance:
    """
    Hellinger distance is an f divergence measure, similar to the Kullback-Leibler (KL) divergence.
    However, unlike KL Divergence the Hellinger divergence is symmetric and bounded over a probability space.
    """

    distrib_t: np.ndarray
    distrib_u: np.ndarray

    def compute(self) -> float:
        return np.sqrt(
            0.5 * ((np.sqrt(self.distrib_u) - np.sqrt(self.distrib_t)) ** 2).sum()
        )


@dataclass
class KullbackLeiblerDivergence:
    """
    KL Divergence (or relative entropy) is a measure of how one probability distribution differs from another.
    It is an asymmetric measure (thus it's not a metric) and it doesn't satisfy the triangle inequality.
    KL Divergence of 0, indicates two identical distributions.
    """

    distrib_t: np.ndarray
    distrib_u: np.ndarray

    def compute(self, capping=None, kld_scaling=0.0001) -> float:
        t_u = np.sum(
            np.where(
                self.distrib_t != 0,
                (self.distrib_t)
                * np.log(
                    self.distrib_t
                    / np.where(self.distrib_u != 0, self.distrib_u, kld_scaling)
                ),
                0,
            )
        )
        u_t = np.sum(
            np.where(
                self.distrib_u != 0,
                (self.distrib_u)
                * np.log(
                    self.distrib_u
                    / np.where(self.distrib_t != 0, self.distrib_t, kld_scaling)
                ),
                0,
            )
        )
        result = t_u + u_t
        if capping:
            return capping if result == float("inf") else result
        return result


class VirtualDrift:
    def __init__(
        self,
        prediction_col: Optional[str] = None,
        label_col: Optional[str] = None,
        feature_weights: Optional[List[float]] = None,
        inf_capping: Optional[float] = 10,
    ):
        self.prediction_col = prediction_col
        self.label_col = label_col
        self.feature_weights = feature_weights
        self.capping = inf_capping
        self.discretizers: Dict[str, KBinsDiscretizer] = {}
        self.metrics = {
            "tvd": TotalVarianceDistance,
            "hellinger": HellingerDistance,
            "kld": KullbackLeiblerDivergence,
        }

    def dict_to_histogram(self, histogram_dict):
        histograms = {}
        for feature, stats in histogram_dict.items():
            histograms[feature] = stats["hist"][0]

        # Get features value counts
        histograms = pd.concat(
            [
                pd.DataFrame(data=hist, columns=[feature])
                for feature, hist in histograms.items()
            ],
            axis=1,
        )
        # To Distribution
        histograms = histograms / histograms.sum()
        return histograms

    def compute_metrics_over_df(self, base_histogram, latest_histogram):
        drift_measures = {}
        for metric_name, metric in self.metrics.items():
            drift_measures[metric_name] = {
                feature: metric(
                    base_histogram.loc[:, feature], latest_histogram.loc[:, feature]
                ).compute()
                for feature in base_histogram
            }
        return drift_measures

    def compute_drift_from_histograms(self, feature_stats, current_stats):
        # Process histogram dictionaries to Dataframe of the histograms
        # with Feature histogram as cols
        base_histogram = self.dict_to_histogram(feature_stats)
        latest_histogram = self.dict_to_histogram(current_stats)

        # Verify all the features exist between datasets
        base_features = set(base_histogram.columns)
        latest_features = set(latest_histogram.columns)
        if not base_features == latest_features:
            raise ValueError(
                f"Base dataset and latest dataset have different featuers: {base_features} <> {latest_features}"
            )

        # Compute the drift per feature
        features_drift_measures = self.compute_metrics_over_df(
            base_histogram.loc[:, base_features], latest_histogram.loc[:, base_features]
        )

        # Compute total drift measures for features
        for metric_name in self.metrics.keys():
            feature_values = list(features_drift_measures[metric_name].values())
            features_drift_measures[metric_name]["total_sum"] = np.sum(feature_values)
            features_drift_measures[metric_name]["total_mean"] = np.mean(feature_values)

            # Add weighted mean by given feature weights if provided
            if self.feature_weights:
                features_drift_measures[metric_name]["total_weighted_mean"] = np.dot(
                    feature_values, self.feature_weights
                )

        drift_result = defaultdict(dict)

        for feature in base_features:
            for metric, values in features_drift_measures.items():
                drift_result[feature][metric] = values[feature]
                sum = features_drift_measures[metric]["total_sum"]
                mean = features_drift_measures[metric]["total_mean"]
                drift_result[f"{metric}_sum"] = sum
                drift_result[f"{metric}_mean"] = mean
                if self.feature_weights:
                    metric_measure = features_drift_measures[metric]
                    weighted_mean = metric_measure["total_weighted_mean"]
                    drift_result[f"{metric}_weighted_mean"] = weighted_mean

        if self.label_col:
            label_drift_measures = self.compute_metrics_over_df(
                base_histogram.loc[:, self.label_col],
                latest_histogram.loc[:, self.label_col],
            )
            for metric, values in label_drift_measures.items():
                drift_result[self.label_col][metric] = values[metric]

        if self.prediction_col:
            prediction_drift_measures = self.compute_metrics_over_df(
                base_histogram.loc[:, self.prediction_col],
                latest_histogram.loc[:, self.prediction_col],
            )
            for metric, values in prediction_drift_measures.items():
                drift_result[self.prediction_col][metric] = values[metric]

        return drift_result

    @staticmethod
    def parquet_to_stats(parquet_path: Path):
        df = pd.read_parquet(parquet_path)
        df = list(df["named_features"])
        df = pd.DataFrame(df)
        latest_stats = DFDataInfer.get_stats(df, InferOptions.Histogram)
        return latest_stats


class BatchProcessor:
    def __init__(self, context: MLClientCtx, project: str):
        self.context = context
        self.project = project
        self.virtual_drift = VirtualDrift(inf_capping=10)

        self.parquet_path = config.model_endpoint_monitoring.store_prefixes.default.format(
            project=self.project, kind="parquet"
        )

        kv_path = config.model_endpoint_monitoring.store_prefixes.default.format(
            project=self.project, kind="endpoints"
        )

        _, self.kv_container, self.kv_path = parse_model_endpoint_store_prefix(kv_path)

        stream_path = config.model_endpoint_monitoring.store_prefixes.default.format(
            project=self.project, kind="log_stream"
        )

        _, self.stream_container, self.stream_path = parse_model_endpoint_store_prefix(
            stream_path
        )

        self.default_possible_drift_threshold = (
            config.model_endpoint_monitoring.drift_thresholds.default.possible_drift
        )
        self.default_drift_detected_threshold = (
            config.model_endpoint_monitoring.drift_thresholds.default.drift_detected
        )

        self.db = get_run_db()
        self.v3io = get_v3io_client()

        logger.info("Initializing BatchProcessor", batch_path=self.parquet_path)

    def post_init(self):
        try:
            self.v3io.stream.create(
                container=self.stream_container, stream_path=self.stream_path
            )
        except Exception as e:
            self.context.logger.error(
                "Failed to initialize log stream",
                container=self.stream_container,
                stream_path=self.stream_path,
                exc=e,
            )

    def run(self):

        try:
            endpoints = self.db.list_endpoints(self.project)
        except Exception as e:
            self.context.logger.error("Failed to list endpoints", exc=e)
            return

        active_endpoints = set()
        for endpoint in endpoints.endpoints:
            if endpoint.spec.active:
                active_endpoints.add(endpoint.metadata.uid)

        store, _ = store_manager.get_or_create_store(self.parquet_path)
        fs = store.get_filesystem(silent=False)

        for endpoint_dir in fs.ls(self.parquet_path):
            endpoint_id = endpoint_dir.split("=")[-1]
            if endpoint_id not in active_endpoints:
                continue

            try:
                last_year = self.get_last_created_dir(fs, endpoint_dir)
                last_month = self.get_last_created_dir(fs, last_year)
                last_day = self.get_last_created_dir(fs, last_month)
                last_hour = self.get_last_created_dir(fs, last_day)

                parquet_files = fs.ls(last_hour)
                last_parquet = sorted(parquet_files, key=lambda k: k["mtime"])[-1]
                parquet_name = last_parquet["name"]

                endpoint = self.db.get_endpoint(
                    project=self.project, endpoint_id=endpoint_id
                )

                current_stats = self.virtual_drift.parquet_to_stats(
                    parquet_path=parquet_name
                )

                drift_result = self.virtual_drift.compute_drift_from_histograms(
                    feature_stats=endpoint.status.feature_stats,
                    current_stats=current_stats,
                )

                drift_status = self.check_for_drift(
                    drift_result=drift_result, endpoint=endpoint
                )

                if drift_status == "POSSIBLE_DRIFT" or drift_status == "DRIFT_DETECTED":
                    self.v3io.stream.put_records(
                        container=self.stream_container,
                        stream_path=self.stream_path,
                        records=[{"drift_status": drift_status, **drift_result}],
                    )

                self.v3io.kv.update(
                    container=self.kv_container,
                    table_path=self.kv_path,
                    key=endpoint_id,
                    attributes={
                        "current_stats": json.dumps(current_stats),
                        "drift_measures": json.dumps(drift_result),
                        "drift_status": drift_status,
                    },
                )

                self.context.logger.info(
                    "Done updating drift measures",
                    endpoint_id=endpoint_id,
                    file=parquet_name,
                )
            except Exception as e:
                self.context.logger.warning(
                    f"Virtual Drift failed for endpoint", endpoint_id=endpoint_id, exc=e
                )

    def check_for_drift(self, drift_result, endpoint):
        tvd_mean = drift_result.get("tvd", {}).get("total_mean")
        hellinger_mean = drift_result.get("hellinger", {}).get("total_mean")

        drift_mean = 0.0
        if tvd_mean and hellinger_mean:
            drift_mean = (tvd_mean + hellinger_mean) / 2

        possible_drift = endpoint.spec.monitor_configuration.get(
            "possible_drift", self.default_possible_drift_threshold
        )
        drift_detected = endpoint.spec.monitor_configuration.get(
            "possible_drift", self.default_drift_detected_threshold
        )

        drift_status = "NO_DRIFT"
        if drift_mean >= drift_detected:
            drift_status = "DRIFT_DETECTED"
        elif drift_mean >= possible_drift:
            drift_status = "POSSIBLE_DRIFT"

        return drift_status

    @staticmethod
    def get_last_created_dir(fs, endpoint_dir):
        dirs = fs.ls(endpoint_dir)
        last_dir = sorted(dirs, key=lambda k: k.split("=")[-1])[-1]
        return last_dir


def handler(context: MLClientCtx, project: str):
    batch_processor = BatchProcessor(context, project)
    batch_processor.post_init()
    batch_processor.run()
