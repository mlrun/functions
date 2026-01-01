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
import pathlib
import tempfile
from difflib import SequenceMatcher

import mlrun
import pytest

expected_outputs = [
    "This is a speech to text test.",
    "In the heart of the stadium, "
    "cheers paint the air as the ball weaves its tale across the pitch. "
    "With each kick, players chase their dreams, guided by the rhythmic dance of teamwork. "
    "The crowd roars, a symphony of passion, "
    "as the game writes its unpredictable story on the field of destiny.",
]
models = [
    "openai/whisper-tiny",
]


@pytest.mark.skipif(os.system("which ffmpeg") != 0, reason="ffmpeg not installed")
@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("audio_path", ["./data", "./data/speech_01.mp3"])
def test_transcribe(model_name: str, audio_path: str):
    # Setting variables and importing function:
    artifact_path = tempfile.mkdtemp()
    project = mlrun.get_or_create_project("test")
    transcribe_function = project.set_function(
        "transcribe.py", "transcribe", kind="job", image="mlrun/mlrun"
    )
    # transcribe_function = mlrun.import_function("function.yaml")
    temp_dir = tempfile.mkdtemp()

    # Running transcribe function:
    transcribe_run = transcribe_function.run(
        handler="transcribe",
        params={
            "data_path": audio_path,
            "model_name": model_name,
            "device": "cpu",
            "output_directory": temp_dir,
        },
        local=True,
        returns=["output_dir: path", "dataset: dataset", "errored_files"],
        artifact_path=artifact_path,
    )

    artifact_path += (
        f"/{transcribe_run.metadata.name}/{transcribe_run.metadata.iteration}/"
    )

    # Getting actual files from run (text and errored):
    input_files = (
        os.listdir(audio_path)
        if pathlib.Path(audio_path).is_dir()
        else [pathlib.Path(audio_path).name]
    )
    expected_text_files = sorted([f for f in input_files if f.endswith("mp3")])
    error_files = list(set(input_files) - set(expected_text_files))
    expected_text_files = [f.replace("mp3", "txt") for f in expected_text_files]
    text_files = sorted(os.listdir(temp_dir))

    # Check that the text files are saved in output_directory:
    assert text_files == expected_text_files

    # Check that the transcribed text was approximately (90%) generated from audio:
    for text_file, expected in zip(text_files, expected_outputs):
        with open(os.path.join(temp_dir, text_file)) as f:
            output = f.readlines()[0]
            ratio = SequenceMatcher(None, expected, output).ratio()
            assert ratio >= 0.9

    # Check that the dataframe is in the correct size:
    df = mlrun.get_dataitem(artifact_path + "dataset.parquet").as_df()
    assert len(df) == len(expected_text_files)

    # Check errored files:
    if isinstance(transcribe_run.outputs["errored_files"], str):
        actual_errored_files = []
    else:
        actual_errored_files = [
            os.path.basename(errored)
            for errored in transcribe_run.outputs["errored_files"].keys()
        ]
    assert actual_errored_files == error_files

    # Check output_dir:
    zip_dir = mlrun.get_dataitem(artifact_path + "output_dir.zip")
    assert zip_dir.kind == "file"
