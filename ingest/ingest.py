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
from typing import Union, List, Dict

import mlrun.feature_store as fs
from mlrun.execution import MLClientCtx
from mlrun.data_types import InferOptions


def ingest(
    context: MLClientCtx,
    featureset: str,
    source: str,
    targets: List[Union[str, Dict]] = None,
    namespace=None,
    infer_options=None,
    run_config: Union[str, Dict] = None,
    spark_context=None,
    overwrite=None,
):
    """Read local DataFrame, file, URL, or source into the feature store
    Ingest reads from the source, run the graph transformations, infers  metadata and stats
    and writes the results to the default of specified targets

    when targets are not specified data is stored in the configured default targets
    (will usually be NoSQL for real-time and Parquet for offline).

    example::

        stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
        stocks = pd.read_csv("stocks.csv")
        df = ingest(stocks_set, stocks, infer_options=fstore.InferOptions.default())

        # for running as remote job
        config = RunConfig(image='mlrun/mlrun').apply(mount_v3io())
        df = ingest(stocks_set, stocks, run_config=config)

        # specify source and targets
        source = CSVSource("mycsv", path="measurements.csv")
        targets = [CSVTarget("mycsv", path="./mycsv.csv")]
        ingest(measurements, source, targets)

    :param context:       MLRun context
    :param featureset:    feature set object or featureset.uri. (uri must be of a feature set that is in the DB,
                          call `.save()` if it's not)
    :param source:        source dataframe or file path
    :param targets:       optional list of data target objects
    :param namespace:     namespace or module containing graph classes
    :param infer_options: schema and stats infer options
    :param run_config:    function and/or run configuration for remote jobs,
                          see :py:class:`~mlrun.feature_store.RunConfig`
    :param spark_context: local spark session for spark ingestion, example for creating the spark context:
                          `spark = SparkSession.builder.appName("Spark function").getOrCreate()`
                          For remote spark ingestion, this should contain the remote spark service name
    :param overwrite:     delete the targets' data prior to ingestion
                          (default: True for non-scheduled ingest - deletes the targets that are about to be ingested.
                                    False for scheduled ingest - does not delete the target)

    """
    # Setting infer_options to default:
    context._parameters["infer_options"] = infer_options or InferOptions.default()

    context.logger.info(f"Calling ingestion task with: {featureset}")

    # ingest called with mlrun_context, feature_set, source and targets passed with context
    # This params here for documentation purposes only
    fs.ingest(
        mlrun_context=context,
        namespace=namespace,
        spark_context=spark_context,
    )
    context.log_result("featureset", featureset)
