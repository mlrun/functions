import pandas as pd
from mlrun import MLClientCtx
import os
import mlrun
import numpy as np
import keras
import json


class RNN_Model_Serving(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model(suffix=".h5")
        self.model = keras.models.load_model(model_file)

    def predict(self, body):
        try:
            """Generate model predictions from sample."""
            feats = np.asarray(body['inputs'])
            result = self.model.predict(feats)
            result = json.dumps(result.tolist())
            return result
        except Exception as e:
            raise Exception("Failed to predict %s" % e)