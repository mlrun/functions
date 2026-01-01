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
import importlib
import io
import logging
import os
import pathlib
import random
import tempfile
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torchaudio
import tqdm

# Get the global logger:
_LOGGER = logging.getLogger()

OPENAI_API_KEY = "OPENAI_API_KEY"
OPENAI_BASE_URL = "OPENAI_API_BASE"
SAMPLE_RATE = 24000


def generate_multi_speakers_audio(
    data_path: str,
    speakers: list[str] | dict[str, int],
    available_voices: list[str],
    engine: str = "openai",
    output_directory: str = None,
    use_gpu: bool | None = None,
    use_small_models: bool | None = None,
    offload_cpu: bool | None = None,
    model: str | None = None,
    speed: float | None = None,
    sample_rate: int = 16000,
    file_format: str = "wav",
    verbose: bool = True,
    bits_per_sample: int | None = None,
) -> tuple[str, pd.DataFrame, dict]:
    """
    Generate audio files from text files.

    :param data_path:           Path to the text file or directory containing the text files to generate audio from.
    :param speakers:            List / Dict of speakers to generate audio for.
                                If a list is given, the speakers will be assigned to channels in the order given.
                                If dictionary, the keys will be the speakers and the values will be the channels.
    :param available_voices:    List of available voices to use for the generation.
                        See here for the available voices for bark engine:
                        https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
                        See here for the available voices for openai engine:
                        https://beta.openai.com/docs/api-reference/speech
    :param engine:              The engine to use for the generation. Select either "bark" or "openai". Default is "openai".
    :param output_directory:    Path to the directory to save the generated audio files to.
    :param use_gpu:             Whether to use the GPU for the generation. Supported only in "bark" engine.
    :param use_small_models:    Whether to use the small models for the generation. Supported only in "bark" engine.
    :param offload_cpu:         To reduce the memory footprint, the models can be offloaded to the CPU after loading.
                                Supported only in "bark" engine.
    :param model:               Which model to use for the generation. Supported only in "openai" engine.
                                Default is "tts-1".
    :param speed:               The speed of the generated audio. Select a value from `0.25` to `4.0`. `1.0` is the default.
    :param sample_rate:         The sampling rate of the generated audio.
    :param file_format:         The format of the generated audio files.
    :param verbose:             Whether to print the progress of the generation.
    :param bits_per_sample:     Changes the bit depth for the supported formats.
                                Supported only in "wav" or "flac" formats.

    :returns:                   A tuple of:
                                - The output directory path.
                                - The generated audio files dataframe.
                                - The errors' dictionary.
    """

    global _LOGGER
    _LOGGER = _get_logger()
    # Get the input text files to turn to audio:
    data_path = pathlib.Path(data_path).absolute()
    text_files = _get_text_files(data_path=data_path)

    # Prepare the speech engine:
    engine = _get_engine(
        engine=engine,
        use_gpu=use_gpu,
        use_small_models=use_small_models,
        offload_cpu=offload_cpu,
        model=model,
        file_format=file_format,
        speed=speed,
    )

    # Check for per channel generation:
    if isinstance(speakers, dict):
        speaker_per_channel = True
        # Sort the given speakers by channels:
        speakers = {
            speaker: channel
            for speaker, channel in sorted(speakers.items(), key=lambda item: item[1])
        }
    else:
        speaker_per_channel = False

    # Prepare the resampling module:
    resampler = torchaudio.transforms.Resample(
        orig_freq=SAMPLE_RATE, new_freq=sample_rate, dtype=torch.float32
    )

    # Prepare the gap between each speaker:
    gap_between_speakers = np.zeros(int(0.5 * SAMPLE_RATE))

    # Prepare the successes dataframe and errors dictionary to be returned:
    successes = []
    errors = {}

    # Create the output directory:
    if output_directory is None:
        output_directory = tempfile.mkdtemp()
    output_directory = pathlib.Path(output_directory)
    if not output_directory.exists():
        output_directory.mkdir(exist_ok=True, parents=True)

    # Start generating audio:
    # Go over the audio files and transcribe:
    for text_file in tqdm.tqdm(
        text_files, desc="Generating", unit="file", disable=not verbose
    ):
        try:
            # Randomize voices for each speaker:
            chosen_voices = {}
            available_voices_copy = available_voices.copy()
            for speaker in speakers:
                voice = random.choice(available_voices_copy)
                chosen_voices[speaker] = voice
                available_voices_copy.remove(voice)
            # Read text:
            with open(text_file) as fp:
                text = fp.read()
            # Prepare a holder for all the generated pieces (if per channel each speaker will have its own):
            audio_pieces = (
                {speaker: [] for speaker in speakers}
                if speaker_per_channel
                else {"all": []}
            )

            # Generate audio per line:
            for line in text.splitlines():
                # Validate line is in correct speaker format:

                if ": " not in line:
                    if verbose:
                        _LOGGER.warning(f"Skipping line: {line}")
                    continue
                # Split line to speaker and his words:
                current_speaker, sentences = line.split(": ", 1)
                # Validate speaker is known:
                if current_speaker not in speakers:
                    raise ValueError(
                        f"Unknown speaker: {current_speaker}. Given speakers are: {speakers}"
                    )
                for sentence in _split_line(line=sentences):
                    # Generate words audio:
                    audio = engine._generate_audio(
                        text=sentence,
                        voice=chosen_voices[current_speaker],
                    )

                    if speaker_per_channel:
                        silence = np.zeros_like(audio)
                        for speaker in audio_pieces.keys():
                            if speaker == current_speaker:
                                audio_pieces[speaker] += [audio, gap_between_speakers]
                            else:
                                audio_pieces[speaker] += [silence, gap_between_speakers]
                    else:
                        audio_pieces["all"] += [audio, gap_between_speakers]
            # Construct a single audio array from all the pieces and channels:

            audio = np.vstack(
                [np.concatenate(audio_pieces[speaker]) for speaker in speakers]
            ).astype(dtype=np.float32)
            # Resample:
            audio = torch.from_numpy(audio)
            audio = resampler(audio)
            # Save to audio file:
            audio_file = output_directory / f"{text_file.stem}.{file_format}"

            torchaudio.save(
                uri=str(audio_file),
                src=audio,
                sample_rate=sample_rate,
                format=file_format,
                bits_per_sample=bits_per_sample,
            )

            # Collect to the successes:
            successes.append([text_file.name, audio_file.name])
        except Exception as exception:
            # Note the exception as error in the dictionary:
            if verbose:
                _LOGGER.warning(f"Error in file: '{text_file.name}'")
            print(exception)
            errors[text_file.name] = str(exception)

    # Construct the translations dataframe:
    successes = pd.DataFrame(
        successes,
        columns=["text_file", "audio_file"],
    )

    # Print the head of the produced dataframe and return:
    if verbose:
        _LOGGER.info(
            f"Done ({successes.shape[0]}/{len(text_files)})\n"
            f"Translations summary:\n"
            f"{successes.head()}"
        )
    return str(output_directory), successes, errors


class SpeechEngine(ABC):
    @abstractmethod
    def _generate_audio(self, text: str, voice: str) -> np.ndarray:
        pass


class BarkEngine(SpeechEngine):
    def __init__(
        self,
        use_gpu: bool = True,
        use_small_models: bool = False,
        offload_cpu: bool = False,
    ):
        try:
            self.bark = importlib.import_module("bark")
        except ImportError:
            raise ImportError(
                "The 'bark' library is required for the BarkEngine. Please install it using 'pip install bark-ai'."
            )

        self.bark.preload_models(
            text_use_gpu=use_gpu,
            text_use_small=use_small_models,
            coarse_use_gpu=use_gpu,
            coarse_use_small=use_small_models,
            fine_use_gpu=use_gpu,
            fine_use_small=use_small_models,
            codec_use_gpu=use_gpu,
            force_reload=offload_cpu,
        )

    def _generate_audio(self, text: str, voice: str) -> np.ndarray:
        # Generate words audio:
        audio = self.bark.generate_audio(
            text,
            history_prompt=voice,
            silent=True,
        )
        return audio


class OpenAIEngine(SpeechEngine):
    def __init__(
        self, model: str = "tts-1", file_format: str = "wav", speed: float = 1.0
    ):
        try:
            self.openai = importlib.import_module("openai")
            self.pydub = importlib.import_module("pydub")
        except ImportError:
            raise ImportError(
                "The 'openai' and 'pydub' libraries are required for the OpenAIEngine. Please install them using 'pip install openai pydub'."
            )

        api_key = os.getenv(OPENAI_API_KEY)
        base_url = os.getenv(OPENAI_BASE_URL)
        # Check if the key is already in the environment variables:
        if not api_key or not base_url:
            try:
                import mlrun

                context = mlrun.get_or_create_ctx(name="context")
                # Check if the key is in the secrets:
                api_key = context.get_secret(OPENAI_API_KEY)
                base_url = context.get_secret(OPENAI_BASE_URL)
            except ModuleNotFoundError:
                raise OSError(
                    f"One or more of the OpenAI required environment variables ('{OPENAI_API_KEY}', '{OPENAI_BASE_URL}') are missing."
                    f"Please set them as environment variables or install mlrun (`pip install mlrun`)"
                    f"and set them as project secrets using `project.set_secrets`."
                )

        self.client = self.openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.file_format = file_format
        self.speed = speed

    def _generate_audio(self, text: str, voice: str) -> np.ndarray:
        # Generate words audio:
        audio = self.client.audio.speech.create(
            model=self.model,
            input=text,
            voice=voice,
            response_format=self.file_format,
            speed=self.speed,
        )
        audio = audio.content
        audio = self._bytes_to_np_array(audio=audio)
        return audio

    def _bytes_to_np_array(self, audio: bytes):
        if self.file_format == "mp3":
            audio_segment = self.pydub.AudioSegment.from_mp3(io.BytesIO(audio))

            # Convert to raw PCM audio data
            samples = audio_segment.get_array_of_samples()

            # Convert to numpy array
            audio_array = np.array(samples)

            # Normalize to float between -1 and 1
            return audio_array.astype(np.float32) / np.iinfo(samples.typecode).max
        else:
            return np.frombuffer(audio, dtype=np.int16) / 32768.0


def _get_engine(engine: str, file_format: str, **kwargs) -> SpeechEngine:
    # eliminate the None values:
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    if engine == "bark":
        return BarkEngine(**kwargs)
    elif engine == "openai":
        return OpenAIEngine(file_format=file_format, **kwargs)
    else:
        raise ValueError(
            f"Unrecognized engine. The parameter `engine` must be either 'bark' or 'openai'. Given: {engine}"
        )


def _get_text_files(
    data_path: pathlib.Path,
) -> list[pathlib.Path]:
    # Check if the path is of a directory or a file:
    if data_path.is_dir():
        # Get all files inside the directory:
        text_files = list(data_path.glob("*.*"))
    elif data_path.is_file():
        text_files = [data_path]
    else:
        raise ValueError(
            f"Unrecognized data path. The parameter `data_path` must be either a directory path or a file path. "
            f"Given: {str(data_path)} "
        )

    return text_files


def _split_line(line: str, max_length: int = 250) -> list[str]:
    if len(line) < max_length:
        return [line]

    sentences = [
        f"{sentence.strip()}." for sentence in line.split(".") if sentence.strip()
    ]

    splits = []
    current_length = len(sentences[0])
    split = sentences[0]
    for sentence in sentences[1:]:
        if current_length + len(sentence) > max_length:
            splits.append(split)
            split = sentence
            current_length = len(sentence)
        else:
            current_length += len(sentence)
            split += " " + sentence
    if split:
        splits.append(split)

    return splits


def _get_logger():
    global _LOGGER
    try:
        import mlrun

        # Check if MLRun is available:
        context = mlrun.get_or_create_ctx(name="mlrun")
        return context.logger
    except ModuleNotFoundError:
        return _LOGGER
