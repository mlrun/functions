# Copyright 2019 Iguazio
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
from cloudpickle import dump, load

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Optional, Union

rom mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.artifacts import TableArtifact, PlotArtifact

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

class FeaturesEngineer(BaseEstimator, TransformerMixin):
    """Engineer features from raw input.
    A standard transformer mixin that can be inserted into a scikit learn Pipeline.
    
    To use, 
    >>> ffg = FeaturesEngineer()
    >>> ffg.fit(X)
    >>> x_transformed = ffg.transform(X)
    or
    >>> ffg = FeaturesEngineer()
    >>> x_transformed = ffg.fit_transform(X)
    
    In a pipeline:
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> transformers = [('feature_gen', FeaturesEngineerFeature()), 
                        ('scaler', StandardScaler())]
    >>> transformer_pipe = Pipeline(transformers)
    """
    def fit(self, X, y=None):
        """fit is unused here, but ANY model can be inserted here
        """
        return self

    def transform(self, X, y=None):
        """Transform raw input data as a preprocessing step, (if fit
        estimates a model, then run transform only after calling fit).
        
        :param X: Raw input features
        
        Returns a DataFrame of features.
        """
        x = X.copy()

        # do some cool feature engineering:here we replace by a N(2,2) series
        m = 2.0
        s = 2.0
        n, f = x.shape

        if type(x) == np.ndarray:
            x[:, f - 1] = np.random.normal(m, s, n)
        else:
            x.values[:, f - 1] = np.random.normal(m, s, n)

        x = x.astype("float")

        return x
    
def features_engineer(
    context: MLClientCtx,
    X: Union[DataItem, str],
    target_path: str = '',
    model_key: str = 'features-model'
    features_key: str = 'features'
):
    """Generate features from an input array
    
    The features model will be saved for reuse in an inference pipeline or when
    testing the model. In addition, the transformed features array is made available
    through the artifact store.
    
    :param context:         the function context
    :param X:               input array
    :param target_path:     destination folder for artifacts
    :param model_key        estimated models are saved under this key in the 
                            artifact store
    :param features_key:    transformed features matrix
    """
    feng = FeaturesEngineer()
    
    X = pd.read_parquet(str(X), engine='pyarrow')
    
    feng.fit(X)
    Xt = feng.transform(X)
    
    filepath = os.path.join(target_path, model_key+'.pkl')
    dump(feng, open(filepath, 'wb'))
    context.log_artifact(model_key, target_path=filepath)
    
    filepath = os.path.join(target_path, features_key+'.pkl')
    dump(feng, open(filepath, 'wb'))
    context.log_artifact(features_key, target_path=filepath)