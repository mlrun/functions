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
import json
from speaker_diarization import (
    _get_clustering_diarizer,
    DiarizationConfig,
    _diarize_single_audio,
    _convert_rttm_to_annotation_df,
)
from nemo.collections.asr.models import ClusteringDiarizer


def test_get_clustering_diarizer():
    # Create temporary manifest file
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as temp_manifest:
        manifest_data = {
            "audio_filepath": "/path/to/audio_file",
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": 2,
            "rttm_filepath": "/path/to/rttm/file",
            "uem_filepath": None,
        }
        json.dump(manifest_data, temp_manifest)
        temp_manifest_path = temp_manifest.name

        # Call the _get_clustering_diarizer function
        diarizer = _get_clustering_diarizer(
            manifest_filepath=temp_manifest_path,
            out_dir="/path/to/out_dir",
            vad_model_path="vad_multilingual_marblenet",
            speaker_embeddings_model_path="titanet_large",
            msdd_model_path="diar_msdd_telephonic",
            audio_filepath="/path/to/audio_file",
            rttm_filepath="/path/to/rttm/file",
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


def test_convert_rttm_to_annotation_df():
    # Sample RTTM content
    sample_rttm_content = """SPEAKER sample_audio 1 2.08 1.12 <NA> <NA> speaker_1 <NA>"""
    print(sample_rttm_content.strip())

    # Create a temporary file and write the sample RTTM content to it
    with tempfile.NamedTemporaryFile(
        suffix=".rttm", mode="w+", delete=True
    ) as temp_rttm_file:
        temp_rttm_file.write(sample_rttm_content.strip())
        temp_rttm_file_path = os.path.dirname(temp_rttm_file.name)
        # Call the function to convert RTTM to annotation DataFrame
        annotation_df, _ = _convert_rttm_to_annotation_df(temp_rttm_file_path)

    # Assertions to check if the DataFrame is formed correctly
    assert annotation_df.shape == (3, 3)
    assert "start" in annotation_df.columns
    assert "end" in annotation_df.columns
    assert "label" in annotation_df.columns
    assert annotation_df.iloc[0].start == 2.08
    assert annotation_df.iloc[0].end == 3.20  # 2.08 + 1.12
    assert annotation_df.iloc[0].label == "speaker_1"
    assert annotation_df.iloc[1].label == "speaker_2"
    assert annotation_df.iloc[2].label == "speaker_1"
