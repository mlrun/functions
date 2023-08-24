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
from speaker_diarization import _get_clustering_diarizer, DiarizationConfig, _diarize_single_audio
from nemo.collections.asr.models import ClusteringDiarizer

def test_get_clustering_diarizer():
    # Create temporary manifest file
    with tempfile.NamedTemporaryFile(suffix=".json", mode='w') as temp_manifest:
        manifest_data = {
            'audio_filepath': '/path/to/audio_file',
            'offset': 0,
            'duration': None,
            'label': 'infer',
            'text': '-',
            'num_speakers': 2,
            'rttm_filepath': '/path/to/rttm/file',
            'uem_filepath': None
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
            rttm_filepath="/path/to/rttm/file"
        )
        # Check if the returned object is of type DiarizationConfig
        assert isinstance(diarizer, ClusteringDiarizer), f"Expected ClusteringDiarizer, but got {type(diarizer)}"

def test_diarize_single_audio():
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_file_path = "./data/real_state.mp3"  # Replace with the path to your mp3 file
        output_dir = temp_dir
        
        # Run the function to be tested
        _diarize_single_audio(audio_file_path, output_dir)
