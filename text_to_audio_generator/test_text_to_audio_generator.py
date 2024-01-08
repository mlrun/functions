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

import mlrun
import tempfile
import pytest


@pytest.mark.parametrize("file_format,bits_per_sample", [("wav", 8), ("mp3", None)])
def test_generate_multi_speakers_audio(file_format, bits_per_sample):
    text_to_audio_generator_function = mlrun.import_function("function.yaml")
    with tempfile.TemporaryDirectory() as test_directory:
        function_run = text_to_audio_generator_function.run(
            handler="generate_multi_speakers_audio",
            inputs={"data_path": "data/test_data.txt"},
            params={
                "output_directory": test_directory,
                "speakers": {"Agent": 0, "Client": 1},
                "available_voices": [
                    "v2/en_speaker_0",
                    "v2/en_speaker_1",
                ],
                "use_small_models": True,
                "use_gpu": False,
                "offload_cpu": True,
                "file_format": file_format,
                "bits_per_sample": bits_per_sample,
            },
            local=True,
            returns=[
                "audio_files: path",
                "audio_files_dataframe: dataset",
                "text_to_speech_errors: file",
            ],
            artifact_path=test_directory,
        )
    assert function_run.error == "Run state (completed) is not in error state"
    for key in ["audio_files", "audio_files_dataframe", "text_to_speech_errors"]:
        assert key in function_run.outputs and function_run.outputs[key] is not None
