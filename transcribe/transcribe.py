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
from typing import Literal, Tuple
from functools import partial

import librosa
import mlrun
import pandas as pd
import whisper
from tqdm.auto import tqdm

from pyannote.audio import Pipeline
import torch
import pyhub


def transcribe(
    context: mlrun.MLClientCtx,
    input_path: str,
    model_name: str = "base",
    device: Literal["cuda", "cpu"] = None,
    decoding_options: dict = None,
    output_directory: str = None,
) -> Tuple[pathlib.Path, pd.DataFrame, dict]:
    """
    Transcribe audio files into text files and collect additional data.
    The end result is a directory of transcribed text files
     and a dataframe containing the following columns:

    * audio_file - The original audio file name.
    * transcription_file - The transcribed text file name in the output directory.
    * language - The detected language in the audio file.
    * length - The length of the audio file.
    * rate_of_speech - The number of words divided by the audio file length.

    :param context:               MLRun context.
    :param input_path:            A directory of the audio files or a single file to transcribe.
    :param output_directory:      Path to a directory to save all transcribed audio files.
    :param model_name:            One of the official model names listed by `whisper.available_models()`.
    :param device:                Device to load the model. Can be one of {"cuda", "cpu"}.
                                  Default will prefer "cuda" if available.
    :param decoding_options:      A dictionary of options to construct a `whisper.DecodingOptions`.

    :returns: A tuple of:

              * Path to the output directory.
              * A dataframe dataset of the transcribed file names.
              * A dictionary of errored files that were not transcribed.
    """
    # Set output directory:
    if output_directory is None:
        output_directory = tempfile.mkdtemp()

    # Load the model:
    context.logger.info(f"Loading whisper model: '{model_name}'")
    model = whisper.load_model(model_name, device=device)
    context.logger.info("Model loaded.")

    # Prepare the dataframe and errors to be returned:
    df = pd.DataFrame(
        columns=[
            "audio_file",
            "transcription_file",
            "language",
            "length",
            "rate_of_speech",
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

    for i, audio_file in enumerate(tqdm(audio_files, desc="Transcribing", unit="file")):
        try:
            transcription, length, rate_of_speech, language = _single_transcribe(
                audio_file=audio_file,
                model=model,
                decoding_options=decoding_options,
            )

        except Exception as exception:
            # Collect the exception:
            context.logger.warn(f"Error in file: '{audio_file}'")
            errors[str(audio_file)] = str(exception)
        else:
            # Write the transcription to file:
            saved_filename = (
                str(audio_file.relative_to(audio_files_path)).split(".")[0]
                if is_dir
                else audio_file.stem
            )
            transcription_file = output_directory / f"{saved_filename}.txt"
            transcription_file.parent.mkdir(exist_ok=True, parents=True)
            with open(transcription_file, "w") as fp:
                fp.write(transcription)

            # Note in the dataframe:
            df.loc[i - len(errors)] = [
                str(audio_file.relative_to(audio_files_path)),
                str(transcription_file.relative_to(output_directory)),
                language,
                length,
                rate_of_speech,
            ]
    # Return the dataframe:
    context.logger.info(f"Done:\n{df.head()}")

    return output_directory, df, errors


def _single_transcribe(
    audio_file: pathlib.Path,
    model: whisper.Whisper,
    decoding_options: dict = None,
) -> Tuple[str, int, float, str]:
    decoding_options = decoding_options or dict()
    # Load the audio:
    audio = whisper.audio.load_audio(file=str(audio_file))
    # Get audio length:
    length = librosa.get_duration(path=audio_file)
    # Transcribe:
    result = model.transcribe(audio=audio, **decoding_options)
    # Unpack the model's result:
    transcription = result["text"]
    language = result.get("language") or decoding_options.get("language", "")

    # Calculate rate of speech (number of words / audio length):
    rate_of_speech = len(transcription.split()) / length

    return transcription, length, rate_of_speech, language


def _build_context(audio_file_path, start_time, end_time):
    """
    Builds a context for the the current speaker's speech for transcribe
    The context should have all the conversation up to the current speaker's speech
    :param audio_file_path: The path to the audio file
    :param start_time: The start time of the current speaker's speech
    :param end_time: The end time of the current speaker's speech
    :returns: A string of the context
    """
    pass



class Diarizator:
    """
    A class for speaker diarization using pyannote-audio.
    """

    def __init__(self, auth_token, **kwargs):
        """
        :param auth_token: The authorization token for the model.
        :param kwargs: Additional arguments to pass to the model.
        """
        if auth_token:
            self.auth_token = auth_token
        elif os.environ.get("HK_ACCESS_TOKEN"):
            self.auth_token = os.environ.get("HK_ACCESS_TOKEN")
        else:
            raise ValueError("auth_token must be provided, or set as an environment variable HK_ACCESS_TOKEN")

        self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token = self.auth_token
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            self.pipeline = self.pipeline.to(torch.device(0))

    def _convert_to_support_format(self, audio_file_path):
        """
        Converts the audio file to wav format. diarization_pipeline expects the following format wav, flac, ogg, mat

        :param audio_file_path: Path to the audio file
        """
        audio_file_obj = pathlib.Path(audio_file_path)
        convert_func_dict = {
                ".mp3": pydub.AudioSegment.from_mp3,
                ".flv": pydub.AudioSegment.from_flv,
                ".mp4": partial(pydub.AudioSegment.from_file, format="mp4"),
                ".wma": partial(pydub.AudioSegment.from_file, format="wma"),
                }
        # Check if the file is already in supported format
        if audio_file_obj.suffix in [".wav", ".flac", ".ogg", ".mat"]:
            return audio_file_path
        else:
            wav_file = tempfile.mkstemp(prefix="converted_audio_", suffix=".wav")
            if audio_file_obj.suffix in convert_func_dict.keys():
                audio_file_obj = convert_func_dict[audio_file_obj.suffix](audio_file_path)
                audio_file_obj.export(wav_file, format="wav")
                return wav_file
            else:
                raise ValueError(f"Unsupported audio format {audio_file_obj.suffix}")


    def _split_audio_by_speaker(audio_file_path):
        """
        Splits the audio based on speaker diarization using pyannote-audio.

        :param audio_file_path: Path to the audio file

        :returns: A list of tuples where each tuple contains:
            * start_time: Start time of the speaker's segment
            * end_time: End time of the speaker's segment
            * audio_chunk: The audio segment for the speaker
        """
        # diarization_pipeline = ...
        
        # speaker_segments = ...
        
        # audio_chunks = ...
        
        # Placeholder return
        return []
