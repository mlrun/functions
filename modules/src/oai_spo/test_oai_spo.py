# Copyright 2025 Iguazio
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


from oai_spo import OaiHub


def test_oai_hub_initialization():
    """Test that OaiHub can be initialized with required parameters."""
    oai_hub = OaiHub(
        project_name="test-project",
        data_dir="./data",
        default_env_file="default.env",
        local_env_file="local.env",
        pipeline_config_path="pipeline_config.yaml",
        default_image="mlrun/mlrun",
        source="s3://test-bucket",
    )
    
    assert oai_hub.project_name == "test-project"
    assert oai_hub.data_dir == "./data"
    assert oai_hub.project is None

