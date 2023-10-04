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
import os
import json
import numpy as np
from cloudpickle import load
import mlrun


class XGBoostModel(mlrun.serving.V2ModelServer):
    def load(self):
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(str(model_file), "rb"))

    def predict(self, body):
        try:
            feats = np.asarray(body["inputs"], dtype=np.float32).reshape(-1, 5)
            result = self.model.predict(feats, validate_features=False)
            return result.tolist()
        except Exception as e:
            raise Exception("Failed to predict %s" % e)