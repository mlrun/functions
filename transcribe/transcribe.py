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
from typing import Literal, Tuple

import mlrun
import pandas as pd
import whisper
from mutagen.mp3 import MP3
from tqdm.auto import tqdm


def transcribe(
    context: mlrun.MLClientCtx,
    audio_files_directory: str,
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
    :param audio_files_directory: A directory of the audio files to transcribe.
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

    decoding_options = decoding_options or dict()

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
    audio_files_directory = pathlib.Path(audio_files_directory).absolute()
    for i, audio_file in enumerate(
        tqdm(list(audio_files_directory.rglob("*.*")), desc="Transcribing", unit="file")
    ):
        try:
            # Load the audio:
            audio = whisper.audio.load_audio(file=str(audio_file))
            # Get audio length:
            length = MP3(audio_file).info.length
            # Transcribe:
            result = model.transcribe(audio=audio, **decoding_options)
            # Unpack the model's result:
            transcription = result["text"]
            language = result["language"] or decoding_options.language
            transcription_file = (
                output_directory
                / f"{str(audio_file.relative_to(audio_files_directory)).split('.')[0]}.txt"
            )
            transcription_file.parent.mkdir(exist_ok=True, parents=True)
            # Write the transcription to file:
            with open(transcription_file, "w") as fp:
                fp.write(transcription)
            # Calculate rate of speech (number of words / audio length):
            rate_of_speech = len(transcription.split()) / length
            # Note in the dataframe:
            df.loc[i - len(errors)] = [
                str(audio_file.relative_to(audio_files_directory)),
                str(transcription_file.relative_to(output_directory)),
                language,
                length,
                rate_of_speech,
            ]
        except Exception as exception:
            # Collect the exception:
            context.logger.warn(f"Error in file: '{audio_file}'")
            errors[str(audio_file)] = str(exception)

    # Return the dataframe:
    context.logger.info(f"Done:\n{df.head()}")

    return output_directory, df, errors
