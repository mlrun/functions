# Copyright 2023 Iguazio
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
import os.path
import tempfile

import mlrun


def test_translate():
    project = mlrun.new_project("test-translate")
    translate_fn = project.set_function("translate.py", "translate", image="mlrun/mlrun")
    input_text = "Ali her gece bir kitap okur."
    expected_translation = "Ali reads a book every night."

    with tempfile.TemporaryDirectory() as test_dir:
        with tempfile.TemporaryDirectory() as data_dir:
            with open(os.path.join(data_dir, "test_tr.txt"), "w") as f:
                f.write(input_text)
            translate_run = translate_fn.run(
                handler="translate",
                inputs={
                    "data_path": data_dir,
                },
                params={
                    "model_name": "Helsinki-NLP/opus-mt-tr-en",
                    "device": "cpu",
                    "output_directory": test_dir,
                },
                local=True,
                returns=[
                    "files: path",
                    "text_files_dataframe: dataset",
                    "errors: dict",
                ],
                artifact_path=test_dir,
            )
            assert translate_run.status.state == "completed"
            with open(os.path.join(test_dir, "test_tr.txt")) as f:
                assert f.read() == expected_translation

