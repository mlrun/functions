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
import tempfile
import mlrun
import pathlib
import pytest
import json
from speaker_diarization import (
    _get_clustering_diarizer,
    _diarize_single_audio,
    _convert_rttm_to_annotation_df,
)
from nemo.collections.asr.models import ClusteringDiarizer


def test_get_clustering_diarizer():
    # Create temporary manifest file
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as temp_manifest:
        manifest_data = {
            "audio_filepath": "./data/real_state.mp3",
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": 2,
            "uem_filepath": None,
        }
        json.dump(manifest_data, temp_manifest)
        temp_manifest_path = temp_manifest.name

        # Call the _get_clustering_diarizer function
        diarizer = _get_clustering_diarizer(
            manifest_filepath=temp_manifest_path,
            out_dir="./output",
            vad_model_path="vad_multilingual_marblenet",
            speaker_embeddings_model_path="titanet_large",
            msdd_model_path="diar_msdd_telephonic",
            audio_filepath="./data/real_state.mp3",
        )
        # Check if the returned object is of type DiarizationConfig
        assert isinstance(
            diarizer, ClusteringDiarizer
        ), f"Expected ClusteringDiarizer, but got {type(diarizer)}"


def test_diarize_single_audio():
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_file_path = (
            "./data/real_state.mp3"  # Replace with the path to your mp3 file
        )
        output_dir = temp_dir

        # Run the function to be tested
        output_dir, _ = _diarize_single_audio(audio_file_path, output_dir)
        list_of_files = os.listdir(output_dir)

        # Check if the output directory contains the expected files
        assert "speaker_outputs" in list_of_files
        assert "pred_rttms" in list_of_files
        assert "manifest_vad_input.json" in list_of_files
        assert "vad_outputs" in list_of_files


@pytest.mark.parametrize("input_path", ["./data/real_state.mp3"])
def test_all_diarize(input_path: str):
    # Setting variables and importing function:
    artifact_path = tempfile.mkdtemp()
    diarize_function = mlrun.import_function("function.yaml")
    temp_dir = tempfile.mkdtemp()

    # Running diarize function:
    diarize_run = diarize_function.run(
        handler="diarize",
        params={
            "input_path": input_path,
            "output_directory": temp_dir,
            "num_speakers": 2,
            "device": "cpu",
        },
        local=True,
        returns=["output_directory: path", "dataset: dataset", "errored_files"],
        artifact_path=artifact_path,
    )

    artifact_path += f"/{diarize_run.metadata.name}/{diarize_run.metadata.iteration}/"

    # Getting actual files from run (csv and errored):
    input_files = (
        os.listdir(input_path)
        if pathlib.Path(input_path).is_dir()
        else [pathlib.Path(input_path).name]
    )
    expected_csv_files = sorted([f for f in input_files if f.endswith("mp3")])
    error_files = list(set(input_files) - set(expected_csv_files))
    expected_csv_files = [f.replace("mp3", "csv") for f in expected_csv_files]
    csv_files = sorted(
        [item for item in os.listdir(artifact_path) if item.endswith("csv")]
    )

    # Check that the csv files are saved in output_directory:
    assert csv_files == expected_csv_files

    # Check that the dataframe is in the correct size:
    df = mlrun.get_dataitem(artifact_path + "dataset.parquet").as_df()
    assert len(df) == len(expected_csv_files)

    # Check errored files:
    if isinstance(diarize_run.outputs["errored_files"], str):
        actual_errored_files = []
    else:
        actual_errored_files = [
            os.path.basename(errored)
            for errored in diarize_run.outputs["errored_files"].keys()
        ]
    assert actual_errored_files == error_files

    # Check output_dir:
    output_directory = mlrun.get_dataitem(artifact_path + "output_directory")
    assert output_directory.kind == "file"
