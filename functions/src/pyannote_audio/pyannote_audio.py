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

import heapq
import logging
import operator
import os
import pathlib
from functools import reduce, wraps
from typing import Any

import pandas as pd
import pyannote.audio
import pyannote.core
import torch
import torchaudio
from tqdm import tqdm

# Get the global logger:
_LOGGER = logging.getLogger()


def _check_mlrun_and_open_mpi() -> tuple["mlrun.MLClientCtx", "mpi4py.MPI.Intracomm"]:
    is_mpi = False
    try:
        import mlrun

        context = mlrun.get_or_create_ctx(name="mlrun")
        is_mpi = context.labels.get("kind", "job") == "mpijob"

        if is_mpi:
            try:
                from mpi4py import MPI

                return context, MPI.COMM_WORLD
            except ModuleNotFoundError as mpi4py_not_found:
                context.logger.error(
                    "To distribute the function using MLRun's 'mpijob' you need to have `mpi4py` package in your "
                    "interpreter. Please run `pip install mpi4py` and make sure you have open-mpi."
                )
                raise mpi4py_not_found
        else:
            return context, None
    except ModuleNotFoundError as module_not_found:
        if is_mpi:
            raise module_not_found
    return None, None


def open_mpi_handler(
    worker_inputs: list[str], root_worker_inputs: dict[str, Any] = None
):
    global _LOGGER

    # Check for MLRun and OpenMPI availability:
    context, comm = _check_mlrun_and_open_mpi()

    # Check if MLRun is available, set the global logger to MLRun's:
    if context:
        _LOGGER = context.logger

    def decorator(handler):
        if comm is None or comm.Get_size() == 1:
            return handler

        @wraps(handler)
        def wrapper(**kwargs):
            # Get the open mpi environment properties:
            size = comm.Get_size()
            rank = comm.Get_rank()

            # Give the correct chunk of the workers inputs:
            for worker_input in worker_inputs:
                input_argument = kwargs[worker_input]
                if input_argument is None:
                    continue
                if isinstance(input_argument, str):
                    input_argument = _get_audio_files(
                        data_path=pathlib.Path(input_argument).absolute()
                    )
                if len(input_argument) < size:
                    raise ValueError(
                        f"Cannot split the input '{worker_input}' of length {len(input_argument)} to {size} workers. "
                        f"Please reduce the amount of workers for this input."
                    )
                even_chunk_size = len(input_argument) // size
                chunk_start = rank * even_chunk_size
                chunk_end = (
                    (rank + 1) * even_chunk_size
                    if rank + 1 < size
                    else len(input_argument)
                )
                context.logger.info(
                    f"Rank #{rank}: Processing input chunk of '{worker_input}' "
                    f"from index {chunk_start} to {chunk_end}."
                )
                if isinstance(input_argument, list):
                    input_argument = input_argument[chunk_start:chunk_end]
                elif isinstance(input_argument, pd.DataFrame):
                    input_argument = input_argument.iloc[chunk_start:chunk_end:, :]
                kwargs[worker_input] = input_argument

            # Set the root worker only arguments:
            if rank == 0 and root_worker_inputs:
                kwargs.update(root_worker_inputs)

            # Run the worker:
            output = handler(**kwargs)

            # Send the output to the root rank (rank #0):
            output = comm.gather(output, root=0)
            if rank == 0:
                # Join the outputs:
                context.logger.info("Collecting data from workers to root worker.")
                diarization_dictionary = reduce(
                    operator.ior, [dia for dia, _ in output], {}
                )
                errors_dictionary = reduce(operator.ior, [err for _, err in output], {})
                return diarization_dictionary, errors_dictionary
            return None

        return wrapper

    return decorator


@open_mpi_handler(worker_inputs=["data_path"], root_worker_inputs={"verbose": True})
def diarize(
    data_path: str | list[str],
    model_name: str = "pyannote/speaker-diarization-3.0",
    access_token: str = None,
    device: str = None,
    speakers_labels: list[str] = None,
    speaker_prefix: str = "speaker_",
    separate_by_channels: bool = False,
    minimum_speakers: int = None,
    maximum_speakers: int = None,
    verbose: bool = False,
) -> tuple[dict[str, list[tuple[float, float, str]]], dict[str, str]]:
    """
    Perform speech diarization on given audio files using pyannote-audio (https://github.com/pyannote/pyannote-audio).
    The end result is a dictionary with the file names as keys and their diarization as value. A diarization is a list
    of tuples: (start, end, speaker_label).

    To use the `pyannote.audio` models you must pass a Huggingface token and get access to the required models. The
    token can be passed in one of the following options:

    * Use the parameter `access_token`.
    * Set an environment variable named "HUGGING_FACE_HUB_TOKEN".
    * If using MLRun, you can pass it as a secret named "HUGGING_FACE_HUB_TOKEN".

    To get access to the models on Huggingface, visit their page. For example, to use the default diarization model set
    in this function ("pyannote/speaker-diarization-3.0"), you need access for these two models:

    * https://huggingface.co/pyannote/segmentation-3.0
    * https://huggingface.co/pyannote/speaker-diarization-3.0

    Note: To control the recognized speakers in the diarization output you can choose one of the following methods:

    * For a known speakers amount, you may set speaker labels via the `speakers_labels` parameter that will be used in
      the order of speaking in the audio (first person speaking be the first label in the list). In addition, you can do
      diarization per channel (setting the parameter `separate_by_channels` to True). Each label will be assigned to a
      specific channel by order (first label to channel 0, second label to channel 1 and so on). Notice, this will
      increase runtime.
    * For unknown speakers amount, you can set the `speaker_prefix` parameter to add a prefix for each speaker number.
      You can also help the diarization by setting the speakers range via the `speakers_amount_range` parameter.

    :param data_path:            A directory of the audio files, a single file or a list of files to transcribe.
    :param model_name:           One of the official diarization model names (referred as diarization pipelines) of
                                 `pyannote.audio` Huggingface page. Default: "pyannote/speaker-diarization-3.0".
    :param access_token:         An access token to pass for using the `pyannote.audio` models. If not provided, it
                                 will be looking for the environment variable "HUGGING_FACE_HUB_TOKEN". If MLRun is
                                 available, it will look for a secret "HUGGING_FACE_HUB_TOKEN".
    :param device:               Device to load the model. Can be one of {"cuda", "cpu"}. Default will prefer "cuda" if
                                 available.
    :param speakers_labels:      Labels to use for the recognized speakers. Default: numeric labels (0, 1, ...).
    :param separate_by_channels: If each speaker is speaking in a separate channel, you can diarize each channel and
                                 combine the result into a single diarization. Each label set in the `speakers_labels`
                                 parameter will be assigned to a specific channel by order.
    :param speaker_prefix:       A prefix to add for the speakers labels. This parameter is ignored if
                                 `speakers_labels` is not None. Default: "speaker".
    :param minimum_speakers:     Set the minimum expected amount of speakers to be in the audio files. This parameter is
                                 ignored if `speakers_labels` is not None.
    :param maximum_speakers:     Set the maximum expected amount of speakers to be in the audio files. This parameter is
                                 ignored if `speakers_labels` is not None.
    :param verbose:              Whether to present logs of a progress bar and errors. Default: True.

    :returns: A tuple of:

              * Speech diarization dictionary.
              * A dictionary of errored files that were not transcribed.
    """
    global _LOGGER

    # Get the input audio files to diarize:
    if isinstance(data_path, str):
        data_path = pathlib.Path(data_path).absolute()
        audio_files = _get_audio_files(data_path=data_path)
    else:  # Should be a list of files.
        audio_files = data_path

    # Get the Huggingface access token:
    access_token = _get_access_token(parameter=access_token)
    if access_token is None:
        raise ValueError(
            "A Huggingface access token must be provided to use `pyannote.audio` models. Access token can be passed "
            "via one of the following options:\n"
            "* Use the parameter `access_token`.\n"
            "* Set an environment variable named 'HUGGING_FACE_HUB_TOKEN'.\n"
            "* If using MLRun, you can pass it as a secret named 'HUGGING_FACE_HUB_TOKEN'."
        )

    # Load the diarization pipeline:
    pipeline = pyannote.audio.Pipeline.from_pretrained(
        checkpoint_path=model_name, use_auth_token=access_token
    )

    # Set the device:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cpu":
        pipeline.to(torch.device(device))

    # Prepare the successes dataframe and errors dictionary to be returned:
    diarizations = {}
    errors = {}

    # Prepare the diarization keyword arguments:
    diarize_kwargs = {}
    if speakers_labels:
        diarize_kwargs["num_speakers"] = len(speakers_labels)
    else:
        if minimum_speakers:
            diarize_kwargs["min_speakers"] = minimum_speakers
        if maximum_speakers:
            diarize_kwargs["max_speakers"] = maximum_speakers

    # Go over the audio files and diarize:
    for audio_file in tqdm(
        audio_files, desc="Diarizing", unit="file", disable=not verbose
    ):
        try:
            # Load audio file:
            audio, sample_rate = torchaudio.load(uri=audio_file, channels_first=True)
            # Get the diarization (if provided):
            diarizations[audio_file.name] = _diarize(
                audio=audio,
                sample_rate=sample_rate,
                pipeline=pipeline,
                speakers_labels=speakers_labels,
                separate_by_channels=separate_by_channels,
                speaker_prefix=speaker_prefix,
                diarize_kwargs=diarize_kwargs,
            )
        except Exception as exception:
            # Note the exception as error in the dictionary:
            if verbose:
                _LOGGER.warning(f"Error in file: '{audio_file.name}'")
            errors[str(audio_file.name)] = str(exception)
            continue

    # Print the head of the produced dataframe and return:
    if verbose:
        _LOGGER.info(f"Done ({len(diarizations)}/{len(audio_files)})\n")
    return diarizations, errors


def _get_audio_files(
    data_path: pathlib.Path,
) -> list[pathlib.Path]:
    # Check if the path is of a directory or a file:
    if data_path.is_dir():
        # Get all files inside the directory:
        audio_files = list(data_path.glob("*.*"))
    elif data_path.is_file():
        audio_files = [data_path]
    else:
        raise ValueError(
            f"Unrecognized data path. The parameter `data_path` must be either a directory path or a file path. "
            f"Given: {str(data_path)} "
        )

    return audio_files


def _get_access_token(parameter: str) -> str:
    # If given as a parameter, return it:
    if parameter:
        return parameter

    # Otherwise, look at the environment variable:
    environment_variable = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if environment_variable:
        return environment_variable

    # Lastly, try look in the set secrets in MLRun:
    secret = None
    try:
        import mlrun

        context = mlrun.get_or_create_ctx(name="mlrun")
        secret = context.get_secret(key="HUGGING_FACE_HUB_TOKEN")
    except ModuleNotFoundError:
        pass

    return secret


def _diarize(
    audio: torch.Tensor,
    sample_rate: int,
    pipeline: pyannote.audio.Pipeline,
    speakers_labels: list[str],
    separate_by_channels: bool,
    speaker_prefix: str,
    diarize_kwargs: dict,
) -> list[tuple[float, float, str]]:
    # If there is no need for separation by channels, we diarize and return:
    if not separate_by_channels:
        # Diarize:
        diarization: pyannote.core.Annotation = pipeline(
            file={"waveform": audio, "sample_rate": sample_rate}, **diarize_kwargs
        )
        # Verify speakers labels (should not fail here as we set `num_speakers=len(speakers_labels)` when inferring
        # through the pipeline):
        if speakers_labels:
            given_speakers = len(speakers_labels)
            found_speakers = len(set(diarization.labels()))
            if given_speakers < found_speakers:
                raise ValueError(
                    f"Not enough `speakers_labels` were given. Got {given_speakers} labels but the diarization "
                    f"recognized {found_speakers} speakers."
                )
        # Return as a diarization list - a sorted list of tuples of start time, end time and a label (the default label
        # returned is "SPEAKER_i" so we take only the index out of it):
        return [
            (
                segment.start,
                segment.end,
                speakers_labels[int(label.split("_")[1])]
                if speakers_labels
                else f"{speaker_prefix}{int(label.split('_')[1])}",
            )
            for segment, track, label in diarization.itertracks(yield_label=True)
        ]

    # Separate to channels and diarize (we expect only one speaker per channel):
    channel_diarizations = [
        _diarize(
            audio=audio[channel].unsqueeze(
                0
            ),  # Take channel and add a channel dimension to it.
            sample_rate=sample_rate,
            pipeline=pipeline,
            speakers_labels=[
                speakers_labels[channel]
            ],  # Take the channel's label only.
            separate_by_channels=False,
            speaker_prefix=speaker_prefix,
            diarize_kwargs={"num_speakers": 1},  # Set to one speaker.
        )
        for channel in range(audio.shape[0])
    ]

    # Merge the channel diarizations into a single sorted list:
    return list(heapq.merge(*channel_diarizations))
