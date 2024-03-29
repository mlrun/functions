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
import pandas as pd
from typing import Optional, List
from sklearn.datasets import make_classification

from mlrun.execution import MLClientCtx


def gen_class_data(
        context: MLClientCtx,
        n_samples: int,
        m_features: int,
        k_classes: int,
        header: Optional[List[str]],
        label_column: Optional[str] = "labels",
        weight: float = 0.5,
        random_state: int = 1,
        key: str = "classifier-data",
        file_ext: str = "parquet",
        sk_params={}
):
    """Create a binary classification sample dataset and save.
    If no filename is given it will default to:
    "simdata-{n_samples}X{m_features}.parquet".

    Additional scikit-learn parameters can be set using **sk_params, please see https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html for more details.

    :param context:       function context
    :param n_samples:     number of rows/samples
    :param m_features:    number of cols/features
    :param k_classes:     number of classes
    :param header:        header for features array
    :param label_column:  column name of ground-truth series
    :param weight:        fraction of sample negative value (ground-truth=0)
    :param random_state:  rng seed (see https://scikit-learn.org/stable/glossary.html#term-random-state)
    :param key:           key of data in artifact store
    :param file_ext:      (pqt) extension for parquet file
    :param sk_params:     additional parameters for `sklearn.datasets.make_classification`
    """
    features, labels = make_classification(
        n_samples=n_samples,
        n_features=m_features,
        weights=weight,
        n_classes=k_classes,
        random_state=random_state,
        **sk_params)

    # make dataframes, add column names, concatenate (X, y)
    X = pd.DataFrame(features)
    if not header:
        X.columns = ["feat_" + str(x) for x in range(m_features)]
    else:
        X.columns = header

    y = pd.DataFrame(labels, columns=[label_column])
    data = pd.concat([X, y], axis=1)

    context.log_dataset(key, df=data, format=file_ext, index=False)
