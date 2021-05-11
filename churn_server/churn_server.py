import numpy as np
from cloudpickle import load
import mlrun


class ChurnModel(mlrun.serving.V2ModelServer):
    def load(self):
        """
        load multiple models in nested folders, churn model only
        """
        clf_model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(str(clf_model_file), "rb"))

        if "cox" in extra_data.keys():
            cox_model_file = extra_data["cox"]
            self.cox_model = load(open(str(cox_model_file), "rb"))
            if "cox/km" in extra_data.keys():
                km_model_file = extra_data["cox/km"]
                self.km_model = load(open(str(km_model_file), "rb"))
        return

    def predict(self, body):
        try:
            # we have potentially 3 models to work with:
            # if hasattr(self, "cox_model") and hasattr(self, "km_model"):
            # hack for now, just predict using one:
            feats = np.asarray(body["instances"], dtype=np.float32).reshape(-1, 23)
            result = self.model.predict(feats, validate_features=False)
            return result.tolist()
            # else:
            #    raise Exception("models not found")
        except Exception as e:
            raise Exception("Failed to predict %s" % e)
