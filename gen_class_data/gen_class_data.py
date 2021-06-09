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
