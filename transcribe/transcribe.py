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
from collections import Counter
import pathlib
import tempfile
from typing import Literal, Tuple, List
from functools import partial

import librosa
import mlrun
import pandas as pd
import whisper
from tqdm.auto import tqdm

from pyannote.audio import Pipeline
from pyannote.core import notebook, Segment, Annotation
import torch
import pydub


def transcribe(
    context: mlrun.MLClientCtx,
    input_path: str,
    model_name: str = "base",
    device: Literal["cuda", "cpu"] = None,
    decoding_options: dict = None,
    output_directory: str = None,
    condition_enhancement: bool = False,
    csv_path: str = None, # path to the csv file that contains the result of speaker diarization
) -> Tuple[pathlib.Path, pd.DataFrame, dict, list]:
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
    :param condition_enhancement: Whether to add the speaker diarization result to add on the transcription.
    

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
            # Get the results of speaker diarization
            res, audio_file_path = diarizator._run(audio_file, num_speakers=2)
            locals()[f"{audio_file}_segments_df"] = diarizator._to_df(res)
            segments = diarizator._split_audio(audio_file_path, res)

            transcription, length, rate_of_speech, language = _single_transcribe(
                audio_file=audio_file_path,
                segments=segments,
                model=model,
                decoding_options=decoding_options,
            )

            # clean up the temporary files
            os.remove(audio_file_path)

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
            file_name = str(audio_file).split("/")[-1].split(".")[0]
            context.log_dataset(
                f"{file_name}_segments_df",
                df=locals()[f"{audio_file}_segments_df"],
                index=False,
                format="csv",
            )

    # Return the dataframe:
    context.logger.info(f"Done:\n{df.head()}")

    return output_directory, df, errors


def _single_transcribe(
    audio_file: pathlib.Path,
    segments: Tuple[int, str, float, float, pathlib.Path],
    model: whisper.Whisper,
    decoding_options: dict = None,
) -> Tuple[str, int, float, str]:
    """
    Transcribe a single audio file with the speaker segmentation
    and return the transcription, length, rate of speech and language.

    :param audio_file:       Path to the audio file.
    :param segments:         A list of tuples of (idx, label, start_time, end_time, file).
    :param model:            A whisper model.
    :param decoding_options: A dictionary of options to construct a `whisper.DecodingOptions`.

    :returns: A tuple of:
            * The transcription.
            * The length of the audio file.
            * The rate of speech.
            * The detected language.
    """
    decoding_options = decoding_options or dict()

    res = []
    langs = []

    for idx, label, start_time, end_time, file in sorted(segments, key=lambda x: x[0]):
        # Load the audio:
        audio = whisper.audio.load_audio(file=str(file))
        # The initail_prompt is the all the previous transcriptions.
        result = model.transcribe(
            audio=audio, **decoding_options, initial_prompt="\n".join(res[: idx + 1])
        )
        # Unpack the model's result:
        transcription = f"{label}  {start_time}  {end_time}: \n {result['text']}"
        langs.append(result.get("language") or decoding_options.get("language", ""))
        res.append(transcription)
        # clean up the temporary files
        os.remove(file)

    # Get audio length:
    length = librosa.get_duration(path=audio_file)
    transcription = "\n".join(res)
    # Calculate rate of speech (number of words / audio length):
    rate_of_speech = len(transcription.split()) / length
    # Get the language:
    language = Counter(langs).most_common(1)[0][0]

    return transcription, length, rate_of_speech, language



def _split_audio(
    audio_file_path: str, annotation: Annotation
) -> List[Tuple[int, str, float, float, str]]:
    """
    Split an audio file based on a pyannote.core.Annotation object.

    :param audio_file_path: path to the audio file to split.
    :param annotation: pyannote.core.Annotation object with diarization results.

    :returns: A list of tuples with the following items:
                * The index of the segment.
                * The label of the speaker.
                * The start time of the segment.
                * The end time of the segment.
                * The path to the temporary audio file.
    """
    audio = pydub.AudioSegment.from_wav(audio_file_path)
    segments = []
    idx = 0
    for segment, _, label in annotation.itertracks(yield_label=True):
        # Convert to milliseconds:
        start_time = segment.start * 1000
        end_time = segment.end * 1000

        # Extract the segment:
        segment_audio = audio[start_time:end_time]
        # Save the segment to a temporary file:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="tmp_", delete=False
        ) as temp_file:
            segment_audio.export(f"{temp_file.name}", format="wav")
        segments.append((idx, label, start_time, end_time, temp_file.name))
        idx += 1

    return segments
