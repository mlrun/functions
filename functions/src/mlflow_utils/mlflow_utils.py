import zipfile
from typing import Any

import mlflow
import pandas as pd
from mlrun.serving.v2_serving import V2ModelServer


class MLFlowModelServer(V2ModelServer):
    """
    MLFlow tracker Model serving class, inheriting the V2ModelServer class for being initialized automatically by the model
    server and be able to run locally as part of a nuclio serverless function, or as part of a real-time pipeline.
    """

    def load(self):
        """
        loads a model that was logged by the MLFlow tracker model
        """
        # Unzip the model dir and then use mlflow's load function
        model_file, _ = self.get_model(".zip")
        model_path_unzip = model_file.replace(".zip", "")

        with zipfile.ZipFile(model_file, "r") as zip_ref:
            zip_ref.extractall(model_path_unzip)

        self.model = mlflow.pyfunc.load_model(model_path_unzip)

    def predict(self, request: dict[str, Any]) -> list:
        """
        Infer the inputs through the model. The inferred data will
        be read from the "inputs" key of the request.

        :param request: The request to the model using xgboost's predict.
                The input to the model will be read from the "inputs" key.

        :return: The model's prediction on the given input.
        """

        # Get the inputs and set to accepted type:
        inputs = pd.DataFrame(request["inputs"])

        # Predict using the model's predict function:
        predictions = self.model.predict(inputs)

        # Return as list:
        return predictions.tolist()
