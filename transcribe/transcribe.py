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

import logging
import operator
import pathlib
from functools import reduce, wraps
from typing import Any, Dict, List, Literal, NamedTuple, Tuple, Union

import faster_whisper
import pandas as pd
from tqdm import tqdm

# Get the global logger:
_LOGGER = logging.getLogger()


def open_mpi_handler(
    worker_inputs: List[str], root_worker_inputs: Dict[str, Any] = None
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
                output_directory = output[0][0]
                dataframe = pd.concat(objs=[df for _, df, _ in output], axis=0)
                errors_dictionary = reduce(
                    operator.ior, [err for _, _, err in output], {}
                )
                return output_directory, dataframe, errors_dictionary
            return None

        return wrapper

    return decorator


def _check_mlrun_and_open_mpi() -> Tuple["mlrun.MLClientCtx", "mpi4py.MPI.Intracomm"]:
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


@open_mpi_handler(worker_inputs=["data_path"], root_worker_inputs={"verbose": True})
def transcribe(
    data_path: Union[str, List[str]],
    output_directory: str,
    model_name: str = "base",
    device: Literal["cuda", "cpu", "auto"] = "auto",
    compute_type: str = "default",
    language: str = None,
    translate_to_english: bool = False,
    speech_diarization: Dict[str, List[Tuple[float, float, str]]] = None,
    audio_duration: bool = False,
    init_kwargs: dict = None,
    transcribe_kwargs: dict = None,
    verbose: bool = False,
) -> Tuple[str, pd.DataFrame, dict]:
    """
    Transcribe audio files into text files and collect additional data. The end result is a directory of transcribed
    text files and a dataframe containing the following columns:

    * audio_file - The audio file path.
    * transcription_file - The transcribed text file name in the output directory.
    * language - The detected language in the audio file.
    * language_probability - The detected language probability.
    * duration - The duration (in seconds) of the audio file (only if `audio_duration` is set to True).

    :param data_path:               A directory of audio files or a single file or a list of files to transcribe.
    :param output_directory:        Path to a directory to save all transcribed audio files.
    :param model_name:              One of the official model names of Whisper: {'tiny.en', 'tiny', 'base.en', 'base',
                                    'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large'} or a
                                    full name of a fine-tuned whisper model from the huggingface hub.
    :param device:                  Device to load the model. Can be one of {"cuda", "cpu"}. Default will prefer "cuda"
                                    if available. To use a specific GPU or more than one GPU, pass the `device_index`
                                    argument via the `init_kwargs`.
    :param compute_type:            The data type to use for computation. For more information, check
                                    https://opennmt.net/CTranslate2/quantization.html. Default: "default" - will use the
                                    default type depending on the device used.
    :param language:                The spoken language to force Whisper the output language. If None, the Whisper model
                                    will automatically predict the output langauge. Default: None.
    :param translate_to_english:    Whether to translate the English post transcription. Default: False.
    :param speech_diarization:      A speech diarization dictionary with the file names to transcribe as keys and their
                                    diarization as value. The diarization is a list of tuples: (start, end, speaker).
                                    The transcription result will be in the following format:
                                    "{speaker}: text text text.". Files with missing diarizations will print a warning.
                                    Pay attention the diarization must be for the entire duration of the audio file (as
                                    long as Whisper is predicting words up until then).
    :param audio_duration:          Whether to include the audio files duration (in seconds). The estimated duration is
                                    from bitrate and may be inaccurate. Default: False.
    :param init_kwargs:             Additional `WhisperModel.__init__` keyword arguments to use.
    :param transcribe_kwargs:       Additional `WhisperModel.transcribe` keyword arguments to use.
    :param verbose:                 Whether to present logs of a progress bar and errors. Default: False.

    :returns: A tuple of:

              * Path to the output directory.
              * A dataframe dataset of the transcribed file names.
              * A dictionary of errored files that were not transcribed.
    """
    global _LOGGER

    # Get the input audio files to transcribe:
    if verbose:
        _LOGGER.info("Collecting audio files.")
    if isinstance(data_path, str):
        data_path = pathlib.Path(data_path).absolute()
        audio_files = _get_audio_files(data_path=data_path)
    else:
        audio_files = data_path
    if verbose:
        _LOGGER.info(f"Collected {len(audio_files)} audio files.")

    # Load the whisper model:
    if verbose:
        _LOGGER.info(f"Loading model '{model_name}' - using device '{device}'.")
    init_kwargs = init_kwargs or {}
    model = faster_whisper.WhisperModel(
        model_size_or_path=model_name,
        device=device,
        compute_type=compute_type,
        **init_kwargs,
    )
    if verbose:
        _LOGGER.info(f"Model loaded successfully.")

    # Prepare the successes dataframe and errors dictionary to be returned:
    successes = []
    errors = {}

    # Create the output directory:
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Prepare the transcribe keyword arguments:
    transcribe_kwargs = transcribe_kwargs or {}
    transcribe_kwargs["language"] = language
    transcribe_kwargs["task"] = "translate" if translate_to_english else "transcribe"

    # Go over the audio files and transcribe:
    for audio_file in tqdm(
        audio_files, desc="Transcribing", unit="file", disable=not verbose
    ):
        try:
            # Transcribe:
            transcription_and_info = _transcribe(
                audio_file=audio_file,
                model=model,
                transcribe_kwargs=transcribe_kwargs,
                speech_diarization=_get_diarization(  # Get the diarization (if provided).
                    speech_diarization=speech_diarization,
                    file_name=audio_file.name,
                    verbose=verbose,
                ),
                audio_duration=audio_duration,
            )
            # Write the transcription to file:
            transcription_file = _save_to_file(
                transcription=transcription_and_info[0],
                file_name=audio_file.stem,
                output_directory=output_directory,
            )
            # Note as a success in the list:
            successes.append(
                [
                    audio_file.name,
                    transcription_file.name,
                    *transcription_and_info[1:],
                ]
            )
        except Exception as exception:
            # Note the exception as error in the dictionary:
            if verbose:
                _LOGGER.warning(f"Error in file: '{audio_file.name}'")
            errors[str(audio_file.name)] = str(exception)
            continue

    # Construct the transcriptions dataframe:
    columns = [
        "audio_file",
        "transcription_file",
        "language",
        "language_probability",
    ]
    if audio_duration:
        columns.append("duration")
    successes = pd.DataFrame(
        successes,
        columns=columns,
    )

    # Print the head of the produced dataframe and return:
    if verbose:
        _LOGGER.info(
            f"Done ({successes.shape[0]}/{len(audio_files)})\n"
            f"Transcriptions summary:\n"
            f"{successes.head()}"
        )
    return str(output_directory), successes, errors


def _get_audio_files(
    data_path: pathlib.Path,
) -> List[pathlib.Path]:
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


class _DiarizationSegment(NamedTuple):
    start: float
    end: float
    speaker: str


def _get_diarization(
    speech_diarization: Dict[str, List[Tuple[float, float, str]]],
    file_name: str,
    verbose: bool,
) -> Union[List[_DiarizationSegment], None]:
    diarization = None
    if speech_diarization is not None:
        diarization = speech_diarization.get(file_name)
        if diarization is None:
            if verbose:
                _LOGGER.warning(
                    f"Missing speech diarization for the audio file '{file_name}'. Continuing transcribing without "
                    f"diarization."
                )
        diarization = [_DiarizationSegment(*segment) for segment in diarization]
    return diarization


def _get_next_diarization_segment(
    word: faster_whisper.transcribe.Word,
    speech_diarization: List[_DiarizationSegment],
    last_chosen_index: int,
) -> int:
    # Get the last chosen diarization segment:
    last_chosen = speech_diarization[last_chosen_index]

    # If the last chosen segment is the last segment, return it:
    if last_chosen_index == len(speech_diarization) - 1:
        return last_chosen_index

    # If the word ends before the last chosen segment:
    if word.end <= last_chosen.start:
        # Then it is still the closest segment
        return last_chosen_index

    # We check if it ends inside the last chosen segment:
    if word.end < last_chosen.end:
        # Then it still is the closest segment
        return last_chosen_index

    # The word ends after the segment, we need to collect all next segments up until the word ends before them:
    possible_segments = [last_chosen_index]
    for i in range(last_chosen_index + 1, len(speech_diarization)):
        if word.end > speech_diarization[i].end:
            possible_segments.append(i)
            continue
        possible_segments.append(i)
        break

    # Check for the most overlapping option:
    best_overlap = 0
    overlapping_segment = None
    for i in possible_segments:
        overlap = 0
        # If the word starts before segment:
        if word.start <= speech_diarization[i].start:
            # If it ends before the segment, there is an overlap from the start of the segment to the end of the word:
            if word.end < speech_diarization[i].end:
                overlap = word.end - speech_diarization[i].start
            else:
                # The word is wrapping the segment, the overlap is the segment's length:
                overlap = speech_diarization[i].end - speech_diarization[i].start
        # The word starts in segment, check if the word ends in it:
        elif word.end < speech_diarization[i].end:
            # The overlap is the word's length:
            overlap = word.end - word.start
        # The word start in segment but ends after it, the overlap is from the word's start to the segment's end:
        else:
            overlap = speech_diarization[i].end - word.start
        # Check for new best overlap:
        if overlap > best_overlap:
            best_overlap = overlap
            overlapping_segment = i
    if overlapping_segment is not None:
        return overlapping_segment

    # If there is no overlapping segment, return the closest segment:
    best_distance = None
    closest_segment = None
    for i in possible_segments:
        distance = (
            word.start - speech_diarization[i].end
            if word.start > speech_diarization[i].end
            else speech_diarization[i].start - word.end
        )
        if best_distance is None or distance < best_distance:
            best_distance = distance
            closest_segment = i
    return closest_segment


def _construct_transcription(
    segments: List[faster_whisper.transcribe.Segment],
    speech_diarization: List[_DiarizationSegment],
) -> str:
    # If there is no diarization, concatenate all segments and return:
    if speech_diarization is None:
        return " ".join([segment.text for segment in segments])

    # There is a diarization, try to match the Whisper model predicted timestamps to the closest diarization segment
    # (closest diarization segment will be the most overlapping with the word, and if there is no overlap, the closest
    # segment to the word):
    diarization_index = 0
    speaker = speech_diarization[diarization_index].speaker
    text = f"{speaker}:"
    for segment in segments:
        for word in segment.words:
            # Get the next diarization segment:
            diarization_index = _get_next_diarization_segment(
                word=word,
                speech_diarization=speech_diarization,
                last_chosen_index=diarization_index,
            )
            # Check if the segment is of the same speaker:
            if speech_diarization[diarization_index].speaker == speaker:
                # Collect the word:
                text += word.word
            else:
                # Append a newline and update the new speaker:
                speaker = speech_diarization[diarization_index].speaker
                text += f"\n{speaker}:{word.word}"

    return text


def _transcribe(
    audio_file: pathlib.Path,
    model: faster_whisper.WhisperModel,
    transcribe_kwargs: dict,
    speech_diarization: List[_DiarizationSegment],
    audio_duration: bool,
) -> Union[Tuple[str, str, float], Tuple[str, str, float, float]]:
    # Transcribe (Segments is a generator, so we cast to list to begin transcription from start to end):
    segments, info = model.transcribe(
        audio=str(audio_file),
        **transcribe_kwargs,
        word_timestamps=speech_diarization is not None,
    )
    segments = list(segments)

    # Check if speech diarization was provided:
    if speech_diarization is None:
        text = "".join([segment.text for segment in segments])
    else:
        text = _construct_transcription(
            segments=segments,
            speech_diarization=speech_diarization,
        )
    text = text.strip()

    # Return the transcription text and the additional information:
    if audio_duration:
        return text.strip(), info.language, info.language_probability, info.duration
    return text.strip(), info.language, info.language_probability


def _save_to_file(
    transcription: str, file_name: str, output_directory: pathlib.Path
) -> pathlib.Path:
    # Prepare the file full path (checking for no duplications):
    transcription_file = output_directory / f"{file_name}.txt"
    i = 1
    while transcription_file.exists():
        i += 1
        transcription_file = output_directory / f"{file_name}_{i}.txt"

    # Make sure all directories are created:
    transcription_file.parent.mkdir(exist_ok=True, parents=True)

    # Write to file:
    with open(transcription_file, "w") as fp:
        fp.write(transcription)

    return transcription_file
