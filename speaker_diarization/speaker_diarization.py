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
import librosa
import mlrun
import pandas as pd
from tqdm.auto import tqdm
import json
import pydub
from pyannote.core import notebook, Segment, Annotation
from functools import partial
from omegaconf import OmegaConf
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal, Tuple
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import (
    rttm_to_labels,
    labels_to_pyannote_object,
)


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
    model_path: str = "titanet_large"  # Path to the speaker embeddings model
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
    out_dir: str,
    rttm_filepath: str = None,
    offset: int = 0,
    duration: Optional[float] = None,
    label: str = "infer",
    text: str = "-",
    num_speakers: int = 2,
    uem_filepath: Optional[str] = None,
    num_workers: int = 1,
    sample_rate: int = 16000,
    batch_size: int = 64,
    device: Optional[
        str
    ] = None,  # Set to 'cuda:1', (default cuda if cuda available, else cpu)
    verbose: bool = True,
    **kwargs,
) -> ClusteringDiarizer:
    """
    Helper function to form the configs to get a diarizer object.
    To override a parameter nested inside a configuration, the pattern is <config_name>__<parameter_name>__<attribute_name>.

    :param num_workers:       Number of workers for computing
    :param sample_rate:       Sample rate of the audio
    :param batch_size:        Batch size for computing
    :param device:            Device to use for computing
    :param verbose:           Whether to print intermediate outputs
    :param manifest_filepath: Path to the manifest file
    :param audio_filepath:    Path to the audio file
    :param rttm_filepath:     Path to the RTTM file if it's a groud truth inference
    :param offset:            Offset value
    :param duration:          Duration of the audio
    :param label:             Label for the audio
    :param text:              Text representation of the audio
    :param num_speakers:      Number of speakers in the audio
    :param uem_filepath:      Path to the UEM file
    :param out_dir:           Directory to store intermediate files and prediction outputs
    :param kwargs:            Additional arguments to override default config values

    :returns: ClusteringDiarizer object
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
    diarization_config_dict.update(asdict(general_config))
    omega_config = OmegaConf.create(diarization_config_dict)

    return ClusteringDiarizer(omega_config)


# Helper function to convert the audio file to 16k single channel wav file
def _convert_to_support_format(audio_file_path: str) -> str:
    """
    Converts the audio file to wav format. ClusteringDiarizer expects signle channel 16k wav file.

    :param audio_file_path:   Path to the audio file
    :returns audio_file_path: Path to the converted audio file
    """
    audio_file_obj = pathlib.Path(audio_file_path)
    read_func_dict = {
        ".mp3": pydub.AudioSegment.from_mp3,
        ".flv": pydub.AudioSegment.from_flv,
        ".mp4": partial(pydub.AudioSegment.from_file, format="mp4"),
        ".wma": partial(pydub.AudioSegment.from_file, format="wma"),
    }
    # Check if the file is already in supported format
    if audio_file_obj.suffix == ".wav":
        audio = AudioSegment.from_wav(audio_file_path, format="wav")
        if audio.channels != 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        audio.export(audio_file_path, format="wav")
        return audio_file_path
    else:
        wav_file = tempfile.mkstemp(prefix="converted_audio_", suffix=".wav")
        if audio_file_obj.suffix in read_func_dict.keys():
            audio = read_func_dict[audio_file_obj.suffix](audio_file_path)
            if audio.channels != 1:
                audio = audio.set_channels(1)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
            audio.export(wav_file[1], format="wav")
            return wav_file[1]
        else:
            raise ValueError(f"Unsupported audio format {audio_file_obj.suffix}")


def _diarize_single_audio(
    audio_file_path: str,
    output_dir: str,
    num_speakers: int = 2,
    vad_model: str = "vad_multilingual_marblenet",
    speaker_embeddings_model: str = "titanet_large",
    msdd_model: str = "diar_msdd_telephonic",
    device: Optional[
        str
    ] = None,  # Set to 'cuda:1', (default cuda if cuda available, else cpu)
    **kwargs,
) -> Tuple[str, str]:
    """
    Diarizes a single audio file and returns the diarization results.

    :param audio_file_path: Path to the audio file
    :param output_dir:      Path to the output directory
    :param num_speakers:    Number of speakers in the audio file
    :param vad_model:       Name of the VAD model to use
    :param speaker_embeddings_model: Name of the speaker embeddings model to use
    :param msdd_model:      Name of the msdd model to use
    :param device:          Device to use for diarization
    :param kwargs:          Additional arguments to pass to the diarizer following the format <config_name>__<parameter_name>__<attribute_name>.

    :returns: Tuple of two paths:
             * path of the result of the pipeline
             * path of the converted audio file

    """
    # Convert the audio file to supported format
    audio_file_path = _convert_to_support_format(audio_file_path)

    # Create temporary manifest file
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as temp_manifest:
        manifest_data = {
            "audio_filepath": audio_file_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers,
            "uem_filepath": None,
        }
        json.dump(manifest_data, temp_manifest)
        temp_manifest_path = temp_manifest.name

        # Call the _get_clustering_diarizer function
        diarizer = _get_clustering_diarizer(
            manifest_filepath=temp_manifest_path,
            out_dir=output_dir,
            vad_model_path=vad_model,
            speaker_embeddings_model_path=speaker_embeddings_model,
            msdd_model_path=msdd_model,
            audio_filepath=audio_file_path,
            device=device,
            **kwargs,
        )

        # Diarize the audio file
        diarizer.diarize()
    return output_dir, audio_file_path


def _convert_rttm_to_annotation_df(output_dir: str) -> Tuple[pd.DataFrame, Annotation]:
    """
    Converts the rttm file to a pyannote.Annotation and a pandas dataframe.

    :param output_dir:   Path to the output directory
    :returns df:         Pandas dataframe containing the diarization results
    :returns annotation: pyannote.Annotation containing the diarization results
    """
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".rttm"):
                rttm_file = os.path.join(root, file)
                break
    pred_labels = rttm_to_labels(rttm_file)
    annotation = labels_to_pyannote_object(pred_labels)
    lst = [item.split(" ") for item in pred_labels]
    lst = [item for item in lst if item]
    df = pd.DataFrame(lst, columns=["start", "end", "speaker"])
    return df, annotation


def diarize(
    context: mlrun.MLClientCtx,
    input_path: str,
    output_directory: str,
    condition_show_plot: bool = False,
    num_speakers: int = 2,
    vad_model: str = "vad_multilingual_marblenet",
    speaker_embeddings_model: str = "titanet_large",
    msdd_model: str = "diar_msdd_telephonic",
    device: Optional[
        str
    ] = None,  # Set to 'cuda:1', (default cuda if cuda available, else cpu)
    **kwargs,
):
    """
    Diarize audio files into speaker segments
    The final result is a directory containing the diarization results in the form csv files, a dataframe that has the mapping with the audio file to the csv files
    and a plot of the diarization results if condition_show_plot is set to True. The dataframe (csv) will have the following columns:

    * start: Start time of the speaker segment
    * end: End time of the speaker segment
    * speaker: Speaker label

    The pandas dataframe will have the following format:

    * original_audio_file: Path to the original audio file
    * result of model: directory of the diarization results
    * converted_audio_file: Path to the converted audio file
    * speaker_segments: Path to the dataitem that has the speaker labels, start, end

    :param context:                  MLRun context
    :param input_path:               A directory of the audio files or a single file to diarize
    :param output_directory:               Path to the output directory this is where nemo will store the diarization results
    :param condition_show_plot:      If set to True, the diarization results will be plotted
    :param num_speakers:             Number of speakers in the audio file
    :param vad_model:                Name of the VAD model to use
    :param speaker_embeddings_model: Name of the speaker embeddings model to use
    :param msdd_model:               Name of the msdd model to use
    :param device:                   Device to use for diarization (default cuda if cuda available, else cpu)
    :param kwargs:                   Additional arguments to pass to the diarizer following the format <config_name>__<parameter_name>__<attribute_name>.

    :returns: A tuple of:
              * Path to the diarization results (pandas dataframe)
              * A dictionary of errored files that were not diarized
    """
    # Set output directory:
    if output_directory is None:
        output_directory = tempfile.mkdtemp()

    # Prepare the dataframe and errors to be returned:
    df = pd.DataFrame(
        columns=[
            "audio_file",
            "diarization_results",
            "converted_audio_file",
            "speaker_segments",
        ]
    )

    errors = {}

    # Create the output directory:
    output_directory = pathlib.Path(output_directory)
    if not output_directory.exists():
        output_directory.mkdir()

    # Go over the audio files and transcribe:
    audio_files_path = pathlib.Path(input_path).absolute()
    is_dir = True
    if audio_files_path.is_dir():
        audio_files = list(audio_files_path.rglob("*.*"))
    elif audio_files_path.is_file():
        is_dir = False
        audio_files = [audio_files_path]
    else:
        raise ValueError(
            f"audio_files {str(audio_files_path)} must be either a directory path or a file path"
        )

    for i, audio_file in enumerate(tqdm(audio_files, desc="Diarizing", unit="file")):
        try:
            output_dir = output_directory / f"{audio_file.stem}_{i}"
            output_dir, converted_audio_file_path = _diarize_single_file(
                audio_file=audio_file,
                output_dir=output_dir,
                num_speakers=num_speakers,
                vad_model=vad_model,
                speaker_embeddings_model=speaker_embeddings_model,
                msdd_model=msdd_model,
                device=device,
                **kwargs,
            )
        except Exception as exception:
            # Collect the exception:
            context.logger.warn(f"Error in file: '{audio_file}'")
            errors[str(audio_file)] = str(exception)
        else:
            # Convert the rttm file to a pandas dataframe and a pyannote.Annotation:
            df, annotation = _convert_rttm_to_annotation_df(output_dir)
            locals()[f"{audio_file.stem}_segments_df"] = df

            context.log_dataset(
                f"{audio_file.stem}_segments_df",
                df=locals()[f"{audio_file.stem}_segments_df"],
                index=False,
                format="csv",
            )

            # Note in the dataframe:
            df.loc[i - len(errors)] = [
                str(audio_file.relative_to(audio_files_path)),
                str(output_dir.relative_to(output_directory)),
                converted_audio_file_path,
                context.get_dataitem(f"{audio_file.stem}_segments_df").artifact_url,
            ]

    # Return the dataframe:
    context.logger.info(f"Done:\n{df.head()}")

    return output_directory, df, errors
