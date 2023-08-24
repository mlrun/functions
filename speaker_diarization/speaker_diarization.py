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

import pathlib
import tempfile
import librosa
import pandas as pd
from tqdm.auto import tqdm
import json
from omegaconf import OmegaConf
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal, Tuple
from nemo.collections.asr.models import ClusteringDiarizer


# Using Nemo to do speaker diarization we need the following steps:
# 1. Voice Activity Detection, which part of the audio is speech and which part is not
# 2. Speaker Embedding is used to extract the features of the speech
# 3. Clustering the embedding get the clusters of speakers
# 4. Multiscale Diarization decoder is used to obtain the speaker profile and estimated number of speakers


@dataclass
class GeneralConfig:
    name: str = "ClusterDiarizer"
    num_workers: int = 1
    sample_rate: int = 16000
    batch_size: int = 64
    device: Optional[
        str
    ] = None  # Set to 'cuda:1', 'cuda:2', etc., for specific devices
    verbose: bool = True


# Parameters for VAD (Voice Activity Detection)
@dataclass
class VADParameters:
    window_length_in_sec: float = 0.15  # Window length in seconds for VAD context input
    shift_length_in_sec: float = (
        0.01  # Shift length in seconds to generate frame-level VAD prediction
    )
    smoothing: str = "median"  # Type of smoothing method (e.g., "median")
    overlap: float = 0.5  # Overlap ratio for overlapped mean/median smoothing filter
    onset: float = 0.1  # Onset threshold for detecting the beginning of speech
    offset: float = 0.1  # Offset threshold for detecting the end of speech
    pad_onset: float = 0.1  # Additional duration before each speech segment
    pad_offset: float = 0  # Additional duration after each speech segment
    min_duration_on: float = 0  # Threshold for small non-speech deletion
    min_duration_off: float = 0.2  # Threshold for short speech segment deletion
    filter_speech_first: bool = True  # Whether to apply a speech-first filter


# Main VAD Configuration
@dataclass
class VADConfig:
    model_path: str = "vad_multilingual_marblenet"  # Path to the VAD model
    external_vad_manifest: Optional[
        str
    ] = None  # Optional path to an external VAD manifest
    parameters: VADParameters = field(
        default_factory=VADParameters
    )  # Nested VAD parameters


# Parameters for Speaker Embeddings
@dataclass
class SpeakerEmbeddingsParameters:
    window_length_in_sec: List[float] = field(
        default_factory=lambda: [1.5, 1.25, 1.0, 0.75, 0.5]
    )  # Window lengths for speaker embeddings
    shift_length_in_sec: List[float] = field(
        default_factory=lambda: [0.75, 0.625, 0.5, 0.375, 0.25]
    )  # Shift lengths for speaker embeddings
    multiscale_weights: List[int] = field(
        default_factory=lambda: [1, 1, 1, 1, 1]
    )  # Weights for each scale
    save_embeddings: bool = True  # Whether to save the speaker embeddings


# Main Speaker Embeddings Configuration
@dataclass
class SpeakerEmbeddingsConfig:
    model_path: str = "nvidia/speakerverification_en_titanet_large"  # Path to the speaker embeddings model
    parameters: SpeakerEmbeddingsParameters = field(
        default_factory=SpeakerEmbeddingsParameters
    )  # Nested speaker embeddings parameters


# Parameters for Clustering
@dataclass
class ClusteringParameters:
    oracle_num_speakers: bool = False  # Whether to use the oracle number of speakers
    max_num_speakers: int = 8  # Maximum number of speakers per recording
    enhanced_count_thres: int = 80  # Threshold for enhanced speaker counting
    max_rp_threshold: float = 0.25  # Max range of p-value search for resegmentation
    sparse_search_volume: int = 30  # Number of values to examine for sparse search
    maj_vote_spk_count: bool = (
        False  # Whether to take a majority vote for speaker count
    )


# Main Clustering Configuration
@dataclass
class ClusteringConfig:
    parameters: ClusteringParameters = field(
        default_factory=ClusteringParameters
    )  # Nested clustering parameters


# Parameters for MSDD (Multiscale Diarization Decoder) Model
@dataclass
class MSDDParameters:
    use_speaker_model_from_ckpt: bool = (
        True  # Whether to use the speaker model from the checkpoint
    )
    infer_batch_size: int = 25  # Batch size for MSDD inference
    sigmoid_threshold: List[float] = field(
        default_factory=lambda: [0.7]
    )  # Sigmoid threshold for binarized speaker labels
    seq_eval_mode: bool = (
        False  # Whether to use oracle number of speakers for sequence evaluation
    )
    split_infer: bool = True  # Whether to split the input audio for inference
    diar_window_length: int = (
        50  # Length of the split short sequence when split_infer is True
    )
    overlap_infer_spk_limit: int = (
        5  # Limit for estimated number of speakers for overlap inference
    )


# Main MSDD Configuration
@dataclass
class MSDDConfig:
    model_path: str = "diar_msdd_telephonic"  # Path to the MSDD model
    parameters: MSDDParameters = field(
        default_factory=MSDDParameters
    )  # Nested MSDD parameters


# Main Diarization Configuration
@dataclass
class DiarizationConfig:
    manifest_filepath: str  # Path to the manifest file
    out_dir: str  # Output directory
    oracle_vad: bool = False  # Whether to use oracle VAD
    collar: float = 0.25  # Collar value for scoring
    ignore_overlap: bool = True  # Whether to ignore overlap segments
    vad: VADConfig = field(default_factory=VADConfig)  # Nested VAD configuration
    speaker_embeddings: SpeakerEmbeddingsConfig = field(
        default_factory=SpeakerEmbeddingsConfig
    )  # Nested speaker embeddings configuration
    clustering: ClusteringConfig = field(
        default_factory=ClusteringConfig
    )  # Nested clustering configuration
    msdd_model: MSDDConfig = field(
        default_factory=MSDDConfig
    )  # Nested MSDD configuration


# helper function to form the configs to get a diarizer object


def _get_clustering_diarizer(
    manifest_filepath: str,
    audio_filepath: str,
    rttm_filepath: str,
    offset: int = 0,
    duration: Optional[float] = None,
    label: str = "infer",
    text: str = "-",
    num_speakers: int = 2,
    uem_filepath: Optional[str] = None,
    out_dir: str = "output_directory",
    num_workers: int = 1,
    sample_rate: int = 16000,
    batch_size: int = 64,
    device: Optional[
        str
    ] = None,  # Set to 'cuda:1', (default cuda if cuda available, else cpu)
    verbose: bool = True,
    **kwargs
) -> ClusteringDiarizer:
    """
    Helper function to form the configs to get a diarizer object.
    To override a parameter nested inside a configuration, the pattern is <config_name>__<parameter_name>__<attribute_name>.

    :param num_workers: Number of workers for computing
    :param sample_rate: Sample rate of the audio
    :param batch_size: Batch size for computing
    :param device: Device to use for computing
    :param verbose: Whether to print intermediate outputs
    :param manifest_filepath: Path to the manifest file
    :param audio_filepath: Path to the audio file
    :param rttm_filepath: Path to the RTTM file
    :param offset: Offset value
    :param duration: Duration of the audio
    :param label: Label for the audio
    :param text: Text representation of the audio
    :param num_speakers: Number of speakers in the audio
    :param uem_filepath: Path to the UEM file
    :param out_dir: Directory to store intermediate files and prediction outputs
    :param kwargs: Additional arguments to override default config values

    :return: DiarizationConfig object
    """
    # Create the general config
    general_config = GeneralConfig(
        num_workers=num_workers,
        sample_rate=sample_rate,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )

    # Create the manifest file
    meta = {
        "audio_filepath": audio_filepath,
        "offset": offset,
        "duration": duration,
        "label": label,
        "text": text,
        "num_speakers": num_speakers,
        "rttm_filepath": rttm_filepath,
        "uem_filepath": uem_filepath,
    }

    with open(manifest_filepath, "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    # Extracting specific kwargs for each config and updating them
    vad_kwargs = {
        k.split("vad__")[1]: v for k, v in kwargs.items() if k.startswith("vad__")
    }
    vad_parameters_kwargs = {
        k.split("vad__parameters__")[1]: v
        for k, v in kwargs.items()
        if k.startswith("vad__parameters__")
    }

    speaker_embeddings_kwargs = {
        k.split("speaker_embeddings__")[1]: v
        for k, v in kwargs.items()
        if k.startswith("speaker_embeddings__")
    }
    speaker_parameters_kwargs = {
        k.split("speaker_embeddings__parameters__")[1]: v
        for k, v in kwargs.items()
        if k.startswith("speaker_embeddings__parameters__")
    }

    clustering_kwargs = {
        k.split("clustering__")[1]: v
        for k, v in kwargs.items()
        if k.startswith("clustering__")
    }
    clustering_parameters_kwargs = {
        k.split("clustering__parameters__")[1]: v
        for k, v in kwargs.items()
        if k.startswith("clustering__parameters__")
    }

    msdd_model_kwargs = {
        k.split("msdd_model__")[1]: v
        for k, v in kwargs.items()
        if k.startswith("msdd_model__")
    }
    msdd_parameters_kwargs = {
        k.split("msdd_model__parameters__")[1]: v
        for k, v in kwargs.items()
        if k.startswith("msdd_model__parameters__")
    }

    # Constructing the nested parameter configs
    vad_parameters = VADParameters(**vad_parameters_kwargs)
    speaker_parameters = SpeakerEmbeddingsParameters(**speaker_parameters_kwargs)
    clustering_parameters = ClusteringParameters(**clustering_parameters_kwargs)
    msdd_parameters = MSDDParameters(**msdd_parameters_kwargs)

    # Constructing the configs using the nested parameter configs
    vad_config = VADConfig(parameters=vad_parameters, **vad_kwargs)
    speaker_embeddings_config = SpeakerEmbeddingsConfig(
        parameters=speaker_parameters, **speaker_embeddings_kwargs
    )
    clustering_config = ClusteringConfig(
        parameters=clustering_parameters, **clustering_kwargs
    )
    msdd_config = MSDDConfig(parameters=msdd_parameters, **msdd_model_kwargs)

    # Constructing the main DiarizationConfig
    diarization_config = DiarizationConfig(
        manifest_filepath=manifest_filepath,
        out_dir=kwargs.get("out_dir", "") or out_dir,
        oracle_vad=kwargs.get("oracle_vad", False),
        collar=kwargs.get("collar", 0.25),
        ignore_overlap=kwargs.get("ignore_overlap", True),
        vad=vad_config,
        speaker_embeddings=speaker_embeddings_config,
        clustering=clustering_config,
        msdd_model=msdd_config,
    )

    diarization_config_dict = {"diarizer": asdict(diarization_config)}
    diarization_config_dict = diarization_config_dict.update(asdict(general_config))
    import pdb
    pdb.set_trace()
    print(diarization_config_dict)
    omega_config = OmegaConf.load(diarization_config_dict)

    return ClusteringDiarizer(omega_config)
