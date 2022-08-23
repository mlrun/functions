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
from mlrun import code_to_function
import os


def test_gen_class_data():
    fn = code_to_function(name='test_gen_class_data',
                          filename="gen_class_data.py",
                          handler="gen_class_data",
                          kind="job"
                          )
    fn.run(params={
            "n_samples": 10_000,
            "m_features": 5,
            "k_classes": 2,
            "header" : None,
            "weight": [0.5, 0.5],
            "sk_params": {"n_informative": 2},
            "file_ext": "csv"}
        ,local=True
        ,artifact_path="artifacts"

        )

    assert(os.path.exists("artifacts/classifier-data.csv"))