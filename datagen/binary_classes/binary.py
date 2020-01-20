n_samp# Copyright 2019 Iguazio
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
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Optional, List, Any
from sklearn.datasets import make_classification

from mlrun.execution import MLClientCtx


def create_binary_classification(
    context: MLClientCtx = None,
    n_samples: int = 100_000,
    m_features: int = 20,
    features_hdr: Optional[List[str]] = None,
    weight: float = 0.50,
    random_state=1,
    filename: Optional[str] = None,
    target_path: str = "",
    key: str = "",
    **sk_params,
):
    """Create a binary classification sample dataset and save.
    If no filename is given it will default to:
    'simdata-{n_samples}X{m_features}.parquet'.
    All of the scikit-learn parameters can be set using **sk_params
    :param context:       function context
    :param n_samples:     number of rows/samples
    :param m_features:    number of cols/features
    :param features_hdr:  header for features array
    :param weight:        fraction of sample (neg)
    :param random_state:  rng seed (see https://scikit-learn.org/stable/glossary.html#term-random-state)
    :param filename:      optional name for stored data file
    :param target_path:   destimation for file
    :param key:           key of data in artifact store
    :param sk_params:     keyword arguments for scikit-learn's 'make_classification'
    Returns filename of created data (includes path).
    """
    # check directories exist and create filename if None:
    os.makedirs(target_path, exist_ok=True)
    if not filename:
        name = f"simdata-{n_samples:0.0e}X{m_features}.parquet".replace("+", "")
        filename = os.path.join(target_path, name)

    features, labels = make_classification(
        n_samples=n_samples,
        n_features=m_features,
        weights=[weight],  # False
        n_classes=2,
        random_state=random_state,
        **sk_params,
    )

    # make dataframes, add column names, concatenate (X, y)
    X = pd.DataFrame(features)
    if not features_hdr:
        X.columns = ["feat_" + str(x) for x in range(m_features)]
    else:
        X.columns = features_hdr

    y = pd.DataFrame(labels, columns=["labels"])
    data = pd.concat([X, y], axis=1)

    pq.write_table(pa.Table.from_pandas(data), filename)
    context.log_artifact(key, target_path=filename)
