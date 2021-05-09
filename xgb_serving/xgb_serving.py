import os
import json
import numpy as np
import xgboost as xgb
from cloudpickle import load
import mlrun


class XGBoostModel(mlrun.runtimes.MLModelServer):
    def load(self):
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(str(model_file), "rb"))

    def predict(self, body):
        try:
            feats = np.asarray(body["instances"], dtype=np.float32).reshape(-1, 5)
            result = self.model.predict(feats, validate_features=False)
            return result.tolist()
        except Exception as e:
            raise Exception("Failed to predict %s" % e)