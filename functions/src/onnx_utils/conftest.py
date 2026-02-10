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
import shutil
import tempfile

import mlrun
import pytest

PROJECT_NAME = "onnx-utils"


@pytest.fixture(scope="session")
def onnx_project():
    """Create/get the MLRun project once per test session."""
    return mlrun.get_or_create_project(PROJECT_NAME, context="./")


@pytest.fixture(autouse=True)
def test_environment(onnx_project):
    """Setup and cleanup test artifacts for each test."""
    artifact_path = tempfile.mkdtemp()
    yield artifact_path
    # Cleanup - only remove files/dirs from the directory containing this test file,
    # never from an arbitrary CWD (which could be the project root).
    test_dir = os.path.dirname(os.path.abspath(__file__))
    for test_output in [
        "schedules",
        "runs",
        "artifacts",
        "functions",
        "model.pt",
        "model.zip",
        "model_modules_map.json",
        "model_modules_map.json.json",
        "onnx_model.onnx",
        "optimized_onnx_model.onnx",
    ]:
        test_output_path = os.path.join(test_dir, test_output)
        if os.path.exists(test_output_path):
            if os.path.isdir(test_output_path):
                shutil.rmtree(test_output_path)
            else:
                os.remove(test_output_path)
    if os.path.exists(artifact_path):
        shutil.rmtree(artifact_path)
