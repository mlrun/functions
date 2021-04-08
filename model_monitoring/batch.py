import json
from asyncio import get_event_loop
from collections import defaultdict
from dataclasses import dataclass, fields
from os import environ
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from mlrun.api.crud.model_endpoints import ModelEndpoints
from mlrun.api.schemas import ModelEndpointList
from mlrun.data_types.infer import DFDataInfer, InferOptions
from mlrun.run import MLClientCtx
from mlrun.utils import logger
from mlrun.utils.v3io_clients import get_v3io_client
from sklearn.preprocessing import KBinsDiscretizer
from v3iofs import V3ioFS

ISO_8061 = "%Y-%m-%d %H:%M:%S.%f"


@dataclass
class Config:
    container: str
    v3io_access_key: str
    v3io_api: str
    time_format: str

    _environment_override: bool = True

    def __post_init__(self):
        if self._environment_override:
            for field in fields(self):
                if field.name.startswith("_"):
                    continue
                val = environ.get(field.name.upper(), self.__dict__[field.name])
                if isinstance(val, str) and val.startswith("$"):
                    val = json.loads(val[1:])
                setattr(self, field.name, val)


config = Config(
    container="projects",
    v3io_access_key="",
    v3io_api="",
    time_format="%Y-%m-%d %H:%M:%S.%f",  # ISO 8061
)


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
        kld_zero_scaling: Optional[float] = 0.001,
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

    def yaml_to_histogram(self, histogram_yaml):
        histograms = {}
        for feature, stats in histogram_yaml.items():
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

    def compute_drift_from_histograms(self, base_histogram_yaml, latest_histogram_yaml):

        # Process histogram yamls to Dataframe of the histograms
        # with Feature histogram as cols
        base_histogram = self.yaml_to_histogram(base_histogram_yaml)
        latest_histogram = self.yaml_to_histogram(latest_histogram_yaml)

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
    def __init__(self, project: str):
        self.project = project
        self.virtual_drift = VirtualDrift(inf_capping=10)
        self.batch_path = "{container}/{project}/model-endpoints/parquet".format(
            container=config.container, project=self.project
        )

        logger.info("Initializing BatchProcessor", batch_path=self.batch_path)

    def run(self):

        fs = V3ioFS(v3io_api=config.v3io_api, v3io_access_key=config.v3io_access_key,)
        loop = get_event_loop()

        endpoints: ModelEndpointList = loop.run_until_complete(
            ModelEndpoints.list_endpoints(
                access_key=environ.get("V3IO_ACCESS_KEY"), project=self.project
            )
        )

        v3io = get_v3io_client()

        endpoint_key_dirs = fs.ls(self.batch_path)
        endpoint_key_dirs = [d["name"] for d in endpoint_key_dirs]

        for endpoint_dir in endpoint_key_dirs:
            endpoint_id = endpoint_dir.split("=")[-1]

            year_dirs = fs.ls(endpoint_dir)
            year_dirs = [d["name"] for d in year_dirs]
            last_year = sorted(year_dirs, key=lambda k: k.split("=")[-1])[-1]

            month_dirs = fs.ls(last_year)
            month_dirs = [d["name"] for d in month_dirs]
            last_month = sorted(month_dirs, key=lambda k: k.split("=")[-1])[-1]

            day_dirs = fs.ls(last_month)
            day_dirs = [d["name"] for d in day_dirs]
            last_day = sorted(day_dirs, key=lambda k: k.split("=")[-1])[-1]

            hour_dirs = fs.ls(last_day)
            hour_dirs = [d["name"] for d in hour_dirs]
            last_hour = sorted(hour_dirs, key=lambda k: k.split("=")[-1])[-1]

            parquet_files = fs.ls(last_hour)
            last_parquet = sorted(parquet_files, key=lambda k: k["mtime"])[-1]
            parquet_name = last_parquet["name"]

            resp = v3io.kv.get(
                container="projects",
                table_path=f"{self.project}/model-endpoints/endpoints",
                key=endpoint_id,
            )

            try:
                base_stats = json.loads(resp.output.item["feature_stats"])
                current_stats = self.virtual_drift.parquet_to_stats(parquet_name)
                drift_result = self.virtual_drift.compute_drift_from_histograms(
                    base_stats, current_stats
                )

                v3io.kv.update(
                    container="projects",
                    table_path=f"{self.project}/model-endpoints/endpoints",
                    key=endpoint_id,
                    attributes={
                        "current_stats": json.dumps(current_stats),
                        "drift_measures": json.dumps(drift_result),
                        "drift_status": "NO_DRIFT",
                    },
                )
                logger.info(
                    "Done updating drift measures",
                    endpoint_id=endpoint_id,
                    file=parquet_name,
                )
            except Exception:
                pass

            pass


def handler(context: MLClientCtx):
    BatchProcessor(environ.get("PROJECT_NAME")).run()
