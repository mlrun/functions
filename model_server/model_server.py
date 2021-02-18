from cloudpickle import load
from typing import List

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from mlrun.runtimes import MLModelServer


class ClassifierModel(MLModelServer):
    def load(self):
        """Load model from storage."""
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(model_file, "rb"))

    def predict(self, body: dict) -> List:
        """Generate model predictions from sample.

        :param body : A dict of observations, each of which is an 1-dimensional feature vector.

        Returns model predictions as a `List`, one for each row in the `body` input `List`.
        """
        try:
            feats = np.asarray(body["instances"])
            result: np.ndarray = self.model.predict(feats)
            resp = result.tolist()
        except Exception as e:
            raise Exception(f"Failed to predict {e}")

        return resp
