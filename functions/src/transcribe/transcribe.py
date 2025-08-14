# Copyright 2024 Iguazio
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
import os
import tempfile
from functools import reduce, wraps
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, NamedTuple, Tuple, Union

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    AutoModelForCausalLM,
    pipeline,
)
from transformers.utils import is_flash_attn_2_available


class BaseTask:
    """
    A task to write the transcription to file.
    """

    def __init__(
        self, audio_file: Path, transcription_output: Union[dict, str], text_file: Path
    ):
        """
        Initialize the task.

        :param audio_file:           Path to the audio file that was transcribed.
        :param transcription_output: The transcription output from the pipeline. String means an exception was raised.
        :param text_file:            Path to the text file to write the transcription to.
        """
        # Store the parameters:
        self._audio_file = audio_file
        self._transcription_output = transcription_output
        self._text_file = text_file

        # Prepare the error variable:
        self._error: str = None

    def do_task(self):
        """
        Try to perform the task storing an error if occurred.
        """
        if isinstance(self._transcription_output, str):
            self._error = self._transcription_output
            return
        try:
            self._do_task()
        except Exception as exception:
            self._error = str(exception)

    def is_failed(self) -> bool:
        """
        Check if the task failed.

        :returns: Whether the task failed.
        """
        return self._error is not None

    def get_result(self) -> Tuple[str, str]:
        """
        Get the result of the task. If the task failed, the error will be returned, otherwise, the result will be the
        text file name.

        :returns: The task's result.
        """
        if self.is_failed():
            return self._audio_file.name, self._error
        return self._audio_file.name, self._text_file.name

    def to_tuple(self) -> Tuple[str, dict]:
        """
        Convert the task to a tuple to reconstruct it later (used for multiprocessing to pass in queue).

        :returns: The converted task.
        """
        return self.__class__.__name__, {
            "audio_file": self._audio_file,
            "transcription_output": self._transcription_output,
            "text_file": self._text_file,
        }

    def _do_task(self):
        """
        Perform the task - write the transcription to the stored file path.
        """
        # Checking for no duplications:
        i = 1
        while self._text_file.exists():
            i += 1
            self._text_file = (
                self._text_file.parent
                / f"{self._text_file.stem.rsplit('_', 1)[0]}_{i}{self._text_file.suffix}"
            )

        # Make sure all directories are created:
        self._text_file.parent.mkdir(exist_ok=True, parents=True)

        # Write to file:
        with open(self._text_file, "w") as fp:
            fp.write(self._transcription_output["text"])


class SpeechDiarizationTask(BaseTask):
    """
    A task to write the transcription to file with respect to a given speech diarization.
    """

    class _DiarizationSegment(NamedTuple):
        """
        A speech diarization segment.
        """

        start: float
        end: float
        speaker: str

    class _WordTimestamp(NamedTuple):
        """
        A word with its start and end timestamps.
        """

        start: float
        end: float
        text: str

    def __init__(
        self,
        audio_file: Path,
        transcription_output: dict,
        text_file: Path,
        speech_diarization: List[Tuple[float, float, str]],
    ):
        """
        Initialize the task.

        :param audio_file:           Path to the audio file that was transcribed.
        :param transcription_output: The transcription output from the pipeline.
        :param text_file:            Path to the text file to write the transcription to.
        :param speech_diarization:   A speech diarization as a list of tuples: (start, end, speaker).
        """
        super().__init__(
            audio_file=audio_file,
            transcription_output=transcription_output,
            text_file=text_file,
        )
        self._speech_diarization = speech_diarization
        self._segments: List[SpeechDiarizationTask._DiarizationSegment] = None
        self._last_chosen_index = 0

    def to_tuple(self) -> Tuple[str, dict]:
        """
        Convert the task to a tuple to reconstruct it later (used for multiprocessing to pass in queue).

        :returns: The converted task.
        """
        task_class, task_kwargs = super().to_tuple()
        return task_class, {
            **task_kwargs,
            "speech_diarization": self._speech_diarization,
        }

    def _do_task(self):
        """
        Perform the task - write the transcription to the stored file path with respect to the given speech diarization.
        """
        # Check if a speech diarization is given, if not, just write the transcription to file:
        if not self._speech_diarization:
            super()._do_task()
            return

        # Cast the chunks to word timestamps tuples:
        words = [
            SpeechDiarizationTask._WordTimestamp(
                start=chunk["timestamp"][0],
                end=chunk["timestamp"][1],
                text=chunk["text"],
            )
            for chunk in self._transcription_output["chunks"]
        ]

        # Cast speech diarization to segments tuples:
        self._segments = [
            SpeechDiarizationTask._DiarizationSegment(*segment)
            for segment in self._speech_diarization
        ]

        # Try to match the Whisper model predicted timestamps to the closest diarization segment (closest diarization
        # segment will be the most overlapping with the word, and if there is no overlap, the closest segment to the
        # word):
        speaker = self._segments[self._last_chosen_index].speaker
        text = f"{speaker}:"
        for word in words:
            # Get the next diarization segment:
            self._get_next_segment(word=word)
            # Check if the segment is of the same speaker:
            if self._segments[self._last_chosen_index].speaker == speaker:
                # Collect the word:
                text += word.text
            else:
                # Append a newline and update the new speaker:
                speaker = self._segments[self._last_chosen_index].speaker
                text += f"\n{speaker}:{word.text}"

        # Update the transcription output with the new text to write it to file:
        self._transcription_output["text"] = text
        super()._do_task()

    def _get_next_segment(
        self,
        word: _WordTimestamp,
    ):
        """
        Get the next diarization segment the given word falls into. The `self._last_chosen_index` will be updated
        accordingly.

        :param word: The word timestamp to match to the next segment.
        """
        # If the last chosen segment is the last segment, return it:
        if self._last_chosen_index == len(self._segments) - 1:
            return

        # Get the last chosen diarization segment:
        last_chosen = self._segments[self._last_chosen_index]

        # None value may appear if the word is the last word in the audio file, or it was split during inference. In
        # that case, we'll set the last segment:
        if word.end is None:
            self._last_chosen_index = len(self._segments) - 1
            return

        # If the word ends before the last chosen segment:
        if word.end <= last_chosen.start:
            # Then it is still the closest segment
            return

        # We check if it ends inside the last chosen segment:
        if word.end < last_chosen.end:
            # Then it still is the closest segment
            return

        # The word ends after the segment, we need to collect all next segments up until the word ends before them:
        possible_segments = [self._last_chosen_index]
        for i in range(self._last_chosen_index + 1, len(self._segments)):
            if word.end > self._segments[i].end:
                possible_segments.append(i)
                continue
            possible_segments.append(i)
            break

        # Check for the most overlapping option:
        best_overlap = 0
        most_overlapping_segment_index = None
        for i in possible_segments:
            # If the word starts before segment:
            if word.start <= self._segments[i].start:
                # If it ends before the segment, there is an overlap from the start of the segment to the end of the
                # word:
                if word.end < self._segments[i].end:
                    overlap = word.end - self._segments[i].start
                else:
                    # The word is wrapping the segment, the overlap is the segment's length:
                    overlap = self._segments[i].end - self._segments[i].start
            # The word starts in segment, check if the word ends in it:
            elif word.end < self._segments[i].end:
                # The overlap is the word's length:
                overlap = word.end - word.start
            # The word start in segment but ends after it, the overlap is from the word's start to the segment's end:
            else:
                overlap = self._segments[i].end - word.start
            # Check for new best overlap:
            if overlap > best_overlap:
                best_overlap = overlap
                most_overlapping_segment_index = i
        if most_overlapping_segment_index is not None:
            self._last_chosen_index = most_overlapping_segment_index
            return

        # If there is no overlapping segment, return the closest segment:
        best_distance = None
        closest_segment_index = None
        for i in possible_segments:
            distance = (
                word.start - self._segments[i].end
                if word.start > self._segments[i].end
                else self._segments[i].start - word.end
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
                closest_segment_index = i
        self._last_chosen_index = closest_segment_index


class SpeechDiarizationPerChannelTask(BaseTask):
    """
    A task to write the transcription to file with respect to a given speech diarization per channel.
    """

    class _WordTimestamp(NamedTuple):
        """
        A word with its start and end timestamps and speaker label (channel the word was taken from).
        """

        start: float
        end: float
        speaker: str
        text: str

    def __init__(self, audio_file: Path, text_file: Path):
        """
        Initialize the task.

        :param audio_file: Path to the audio file that was transcribed.
        :param text_file:  Path to the text file to write the transcription to.
        """
        super().__init__(
            audio_file=audio_file, transcription_output={}, text_file=text_file
        )
        self._transcription_output_channels: List[Tuple[str, dict]] = []

    @property
    def transcription_output_channels(self) -> List[Tuple[str, dict]]:
        """
        Get the transcription output channels.

        :returns: The transcription output channels.
        """
        return self._transcription_output_channels

    def do_task(self):
        """
        Try to perform the task storing an error if occurred.
        """
        for _, channel_output in self._transcription_output_channels:
            if isinstance(channel_output, str):
                self._error = self._transcription_output_channels
                return
        super().do_task()

    def to_tuple(self) -> Tuple[str, dict]:
        """
        Convert the task to a tuple to reconstruct it later (used for multiprocessing to pass in queue).

        :returns: The converted task.
        """
        task_class, task_kwargs = super().to_tuple()
        task_kwargs.pop("transcription_output")
        return task_class, task_kwargs

    def _do_task(self):
        """
        Perform the task - write the transcription to the stored file path with respect to the given speech diarization
        per channel.
        """
        # Cast the chunks to word timestamps tuples:
        words_per_channel = [
            [
                SpeechDiarizationPerChannelTask._WordTimestamp(
                    start=chunk["timestamp"][0],
                    end=chunk["timestamp"][1],
                    speaker=speaker,
                    text=chunk["text"],
                )
                for chunk in output["chunks"]
            ]
            for speaker, output in self._transcription_output_channels
        ]

        # Merge and sort the words per channel by their start time:
        words = operator.add(*words_per_channel)
        words.sort()

        # Write the transcription to file:
        current_speaker = words[0].speaker
        text = f"{current_speaker}:"
        for word in words:
            # Check if the word's speaker is different from the current one:
            if word.speaker != current_speaker:
                # Append a newline and update the new speaker:
                current_speaker = word.speaker
                text += f"\n{current_speaker}:"
            # Collect the word:
            text += word.text

        # Update the transcription output with the new text to write it to file:
        self._transcription_output["text"] = text
        super()._do_task()


class BatchProcessor:
    """
    A batch processor to process batches of transcriptions. The batch processor is creating tasks and is aimed to be
    working along the transcriber. It can be used with multiprocessing queue or run the tasks directly using the
    associated methods.
    """

    def __init__(self, audio_files: List[Path], output_directory: Path):
        """
        Initialize the batch processor.

        :param audio_files:      The list of all audio files to transcribe.
        :param output_directory: The output directory to write the transcriptions to.
        """
        # Store the parameters:
        self._audio_files = audio_files
        self._output_directory = output_directory

        # Prepare the batching variables:
        self._current_file_index = 0
        self._tasks: List[BaseTask] = []
        self._results: List[Tuple[bool, Tuple[str, str]]] = []

    def process_batch(self, batch: List[Union[dict, str]]):
        """
        Process a batch of transcriptions. Tasks related to the given batch will be created and stored in the batch
        processor.

        :param batch: The batch of transcriptions to process.
        """
        # Get the relevant files belongs to the given batch:
        current_files = self._get_current_files(batch_size=len(batch))

        # Build the diarization tasks:
        self._tasks.extend(
            [
                BaseTask(
                    audio_file=file,
                    transcription_output=batch[i],
                    text_file=self._output_directory / f"{file.stem}.txt",
                )
                for i, file in enumerate(current_files)
            ]
        )

    def get_tasks(self) -> List[BaseTask]:
        """
        Get the tasks to perform.

        :returns: The tasks to perform.
        """
        tasks = self._tasks
        self._tasks = []
        return tasks

    def do_tasks(self):
        """
        Perform the tasks. Should be used if no multiprocessing queue is given to a transcriber.
        """
        for task in self.get_tasks():
            task.do_task()
            self._results.append((task.is_failed(), task.get_result()))

    def get_results(self) -> List[Tuple[bool, Tuple[str, str]]]:
        """
        Get the results of the tasks. The stored results are then cleared.

        :returns: The results of the tasks.
        """
        results = self._results
        self._results = []
        return results

    def _get_current_files(self, batch_size: int) -> List[Path]:
        """
        Get the current files to process.

        :param batch_size: The batch size to progress the current file index.

        :returns: The current files to process.
        """
        end_index = (
            self._current_file_index + batch_size
            if self._current_file_index + batch_size < len(self._audio_files)
            else len(self._audio_files)
        )
        current_files = self._audio_files[self._current_file_index : end_index]
        self._current_file_index = end_index
        return current_files


class SpeechDiarizationBatchProcessor(BatchProcessor):
    """
    A batch processor to process batches of transcriptions with respect to a given speech diarization. The batch
    processor is creating tasks and is aimed to be working along the transcriber. It can be used with multiprocessing
    queue or run the tasks directly using the associated methods.
    """

    def __init__(
        self, audio_files: List[Path], output_directory: Path, speech_diarization: dict
    ):
        """
        Initialize the batch processor.

        :param audio_files:        The list of all audio files to transcribe.
        :param output_directory:   The output directory to write the transcriptions to.
        :param speech_diarization: A speech diarization dictionary to pass along with each processed batch.
        """
        super().__init__(audio_files=audio_files, output_directory=output_directory)
        self._speech_diarization = speech_diarization
        self._audio_files = audio_files

    def process_batch(self, batch: List[dict]):
        """
        Process a batch of transcriptions. Tasks related to the given batch will be created and stored in the batch
        processor.

        :param batch: The batch of transcriptions to process.
        """
        # Get the relevant files belongs to the given batch:
        current_files = self._get_current_files(batch_size=len(batch))

        # Build the diarization tasks:
        self._tasks.extend(
            [
                SpeechDiarizationTask(
                    audio_file=file,
                    transcription_output=batch[i],
                    text_file=self._output_directory / f"{file.stem}.txt",
                    speech_diarization=self._speech_diarization.get(file.name),
                )
                for i, file in enumerate(current_files)
            ]
        )


class PerChannelSpeechDiarizationBatchProcessor(BatchProcessor):
    """
    A batch processor to process batches of transcriptions per channel. The batch processor is creating tasks with the
    selected amount of channels given and is aimed to be working along the transcriber. It can be used with
    multiprocessing queue or run the tasks directly using the associated methods.
    """

    def __init__(
        self,
        audio_files: List[Path],
        output_directory: Path,
        n_channels: int,
        speakers: List[str],
    ):
        """
        Initialize the batch processor.

        :param audio_files:      The list of all audio files to transcribe.
        :param output_directory: The output directory to write the transcriptions to.
        :param n_channels:       The number of channels in each audio file to transcribe.
        :param speakers:         The speakers labels to use for each channel.
        """
        super().__init__(audio_files=audio_files, output_directory=output_directory)

        # Store the parameters:
        self._n_channels = n_channels
        self._speakers = speakers

        # Prepare a channel buffer to store the channels until the current task created is fully covered:
        self._task_in_process: SpeechDiarizationPerChannelTask = None

    def process_batch(self, batch: List[dict]):
        """
        Process a batch of transcriptions. Tasks related to the given batch will be created and stored in the batch
        processor.

        :param batch: The batch of transcriptions to process.
        """
        # Go over the batch and create the tasks:
        for output in batch:
            # Check if there is a task in process:
            if not self._task_in_process:
                # Create a new task:
                self._task_in_process = SpeechDiarizationPerChannelTask(
                    audio_file=self._audio_files[self._current_file_index],
                    text_file=self._output_directory
                    / f"{self._audio_files[self._current_file_index].stem}.txt",
                )
            # Get the channel's speaker:
            speaker = self._speakers[
                len(self._task_in_process.transcription_output_channels)
            ]
            # Collect the channel into the processed task:
            self._task_in_process.transcription_output_channels.append(
                (speaker, output)
            )
            # Check if the task is fully covered (all channels are collected):
            if (
                len(self._task_in_process.transcription_output_channels)
                == self._n_channels
            ):
                # Collect the task and reset the task in process:
                self._tasks.append(self._task_in_process)
                self._current_file_index += 1
                self._task_in_process = None


class Transcriber:
    """
    A transcription wrapper for the Huggingface's ASR pipeline -
    https://huggingface.co/transformers/main_classes/pipelines.html#transformers.AutomaticSpeechRecognitionPipeline to
    use with OpenAI's Whisper models - https://huggingface.co/openai.
    """

    def __init__(
        self,
        model_name: str,
        device: str = None,
        use_flash_attention_2: bool = None,
        use_better_transformers: bool = None,
        assistant_model: str = None,
        max_new_tokens: int = 128,
        chunk_length_s: int = 30,
        batch_size: int = 2,
        spoken_language: str = None,
        translate_to_english: bool = False,
        return_timestamps: Union[bool, Literal["word"]] = False,
        per_channel_transcription: int = 0,
    ):
        """
        Initialize the transcriber.

        :param model_name:                The model name to use. Should be a model from the OpenAI's Whisper models for
                                          best results (for example "tiny", "base", "large", etc.).
        :param device:                    The device to use for inference. If not given, will use GPU if available.
        :param use_flash_attention_2:     Whether to use the Flash Attention 2 implementation. It can be used only with
                                          one of the following GPUs: Nvidia H series and Nvidia A series. T4 support
                                          will be available soon.

                                          Note: If both `use_flash_attention_2` and
                                          `use_better_transformers` are `None`, the optimization will be chosen
                                          automatically according to the available resources.

        :param use_better_transformers:   Whether to use the Better Transformers library to further optimize the model.
                                          Should be used for all use cases that do not support flash attention 2.

                                          Note: If both `use_flash_attention_2` and `use_better_transformers` are
                                          `None`, the optimization will be chosen automatically according to the
                                          available resources.
       :param assistant_model:           The assistant model name to use for inference. Notice that the optimizations
                                          (flash attention 2 and better transformers) will be applied for the assistant
                                          as well. Should be a model from Huggingface's distil-whisper (see here for
                                          more information: https://github.com/huggingface/distil-whisper).
        :param max_new_tokens:            The maximum number of new tokens to generate. This is used to limit the
                                          generation length. Default is 128 tokens.
        :param chunk_length_s:            The audio chunk to split the audio to (in seconds). Default is 30 seconds.
        :param batch_size:                The batch size to use for inference. Default is 2.
        :param spoken_language:           Aim whisper to know what language is spoken. If None, it will try to detect it
                                          for each chunk.
        :param translate_to_english:      Whether to translate the transcriptions to English. Default is False.
        :param return_timestamps:         Whether to return the timestamps of the words. If "word", will return the
                                          timestamps of each word. If True will return the timestamps of each chunk.
                                          Default is False. Aimed to be used for speech diarization.
        :param per_channel_transcription: Whether to do per channel transcription. If needed to run per channel
                                          transcription, pass the number of channels expected for each audio file here.
                                          0 means regular transcription (merge channels).

                                          Note: If `per_channel_transcription` is not 0, `batch_size` must be treated to
                                          be the number of channels and not audio files. Aimed to be used for per
                                          channel speech diarization.
        """
        # Store loading parameters:
        self._model_name = model_name
        self._device = device
        self._use_flash_attention_2 = use_flash_attention_2
        self._use_better_transformers = use_better_transformers
        self._max_new_tokens = max_new_tokens
        self._chunk_length_s = chunk_length_s
        self._batch_size = batch_size
        self._return_timestamps = return_timestamps
        self._per_channel_transcription = per_channel_transcription

        # Store generation parameters:
        self._assistant_model = assistant_model
        self._spoken_language = spoken_language
        self._translate_to_english = translate_to_english

        # Prepare the transcription objects:
        self._transcription_pipeline: AutomaticSpeechRecognitionPipeline = None
        self._generate_kwargs: dict = None

    def load(self):
        """
        Load the transcriber. Must be called before transcribing.
        """
        # Set the device and data type to use (prefer GPU if available):
        device = torch.device(
            self._device or "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

        # Choose the optimization to use (in case the user did not specify any):
        if (
            self._use_flash_attention_2 is None
            and self._use_better_transformers is None
        ):
            # Prefer to use flash attention 2 if available and cuda device is supported (see GPU names to architecture
            # here: https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units#Tesla):
            if device.type == "cuda" and is_flash_attn_2_available():
                cuda_device_name = torch.cuda.get_device_properties(device).name
                if any(
                    cuda_device_name.startswith(gpu_name)
                    for gpu_name in [
                        "NVIDIA A",  # For Ampere architecture (e.g. A10, A30, A100)
                        "NVIDIA H",  # For Hopper architecture (e.g. H100)
                        "NVIDIA L",  # For Ada Lovelace architecture (e.g. L4, L40)
                        "NVIDIA RTX 30",  # For Ada Lovelace architecture (RTX 30 series)
                        "NVIDIA RTX 40",  # For Ada Lovelace architecture (RTX 40 series)
                        "NVIDIA RTX 50",  # For Ada Lovelace architecture (RTX 50 series)
                        # Will be supported soon according to FlashAttention GitHub repo:
                        # https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
                        # "NVIDIA T4",  # For Turing architecture (only T4)
                        # "NVIDIA RTX 20",  # For Turing architecture (RTX 20 series)
                    ]
                ):
                    self._use_flash_attention_2 = True
                else:
                    self._use_better_transformers = True
            else:
                self._use_better_transformers = True

        # Build the optimizations kwargs:
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        if self._use_flash_attention_2:
            if _LOGGER:
                _LOGGER.info(
                    "Using FlashAttention2 optimization - make sure the `flash-attn` package is installed via "
                    "`pip install -U flash-attn --no-build-isolation`"
                )
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif self._use_better_transformers:
            if _LOGGER:
                _LOGGER.info(
                    "Using BetterTransformers optimization - make sure the `optimum` package is installed via "
                    "`pip install -U optimum`"
                )
            model_kwargs["attn_implementation"] = "sdpa"

        # Initialize the speech recognition pipeline:
        self._transcription_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self._model_name,
            model_kwargs=model_kwargs.copy(),
            batch_size=self._batch_size,
            max_new_tokens=self._max_new_tokens,
            chunk_length_s=self._chunk_length_s,
            return_timestamps=self._return_timestamps,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Prepare the generation kwargs:
        self._generate_kwargs = {
            "language": self._spoken_language,
            "task": "translate" if self._translate_to_english else "transcribe",
        }

        # Initialize the assistant model (if needed):
        if self._assistant_model:
            assistant_model = AutoModelForCausalLM.from_pretrained(
                self._assistant_model, torch_dtype=torch_dtype, **model_kwargs
            )
            assistant_model.to(device)
            self._generate_kwargs["assistant_model"] = assistant_model

    def transcribe(
        self,
        audio_files: List[Path],
        batch_processor: BatchProcessor = None,
        batches_queue: Queue = None,
        verbose: bool = False,
    ) -> Union[List[List[dict]], None]:
        """
        Transcribe the given audio files. The transcriptions will be sent to a queue or a batch processor for further
        processing like writing to text files. If no queue or batch processor is given, the transcriptions outputs from
        the pipeline will be returned. Otherwise, `None` is returned.

        :param audio_files:     The audio files to transcribe.
        :param batch_processor: A batch processor.
        :param batches_queue:   A multiprocessing queue to put the batches in.
        :param verbose:         Whether to show a progress bar. Default is False.

        :returns: The transcriptions outputs from the pipeline if no queue or batch processor is given, otherwise,
                  `None`.
        """
        # Wrap the audio files with a function to iterate over them via a generator (save memory and runtime with
        # Huggingface's pipelines as they preload each input while inference is running):
        def audio_iterator() -> Generator[Union[dict, str], None, None]:
            if self._per_channel_transcription:
                for audio_file in audio_files:
                    audio, sampling_rate = torchaudio.load(str(audio_file))
                    audio = audio.numpy()
                    for channel in audio:
                        yield {"raw": channel, "sampling_rate": sampling_rate}
            else:
                for audio_file in audio_files:
                    yield str(audio_file)

        # Create a batch iterator:
        def batch_iterator() -> Generator[List[Union[dict, str]], None, None]:
            batch = []
            for audio in audio_iterator():
                batch.append(audio)
                if len(batch) == self._batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        # Prepare the successes dataframe and errors dictionary to be returned:
        outputs = []

        # Infer through the pipeline:
        for input_batch in tqdm(
            batch_iterator() if self._batch_size > 1 else audio_iterator(),
            desc="Transcribing",
            unit="channel" if self._per_channel_transcription else "audio file",
            total=(
                (
                    (len(audio_files) // self._batch_size)
                    + (len(audio_files) % self._batch_size != 0)
                )
                * (self._per_channel_transcription or 1)
            ),
            disable=not verbose,
        ):
            # Infer:
            try:
                output_batch = self._transcription_pipeline(
                    input_batch,
                    generate_kwargs=self._generate_kwargs,
                )
            except Exception as exception:
                # Collect the exception:
                output_batch = str(exception)
                # Align to batch size:
                output_batch = (
                    [output_batch] * len(input_batch)
                    if isinstance(input_batch, list)
                    else [output_batch]
                )
            # To align with batching, if batch size is 1, wrap the output with a list:
            if isinstance(output_batch, dict):
                output_batch = [output_batch]
            # If a batch processor is given, process the batch:
            if batch_processor:
                # Process it directly:
                batch_processor.process_batch(batch=output_batch)
                batch_processor.do_tasks()
            elif batches_queue:
                # Otherwise, queue the batch:
                batches_queue.put(output_batch)
            else:
                # Otherwise, collect the output as is without processing:
                outputs.append(output_batch)

        # Check if given a multiprocessing queue or a batch processor:
        if batches_queue:
            batches_queue.put(_MULTIPROCESSING_STOP_MARK)

        return outputs if not batch_processor else None


#: The value to send into multiprocessing queues to stop the process:
_MULTIPROCESSING_STOP_MARK = "STOP"


def _multiprocessing_process_batches(
    batch_processor: BatchProcessor,
    batches_queue: Queue,
    tasks_queue: Queue,
    n_task_completers: int,
):
    """
    Process the batches in the given batches queue and put the tasks in the given tasks queue. The function will stop
    when the given batches queue will receive the stop mark. It is aimed to be used with multiprocessing as a process.

    :param batch_processor:   A batch processor to process the batches.
    :param batches_queue:     A queue to get the batches from.
    :param tasks_queue:       A queue to put the tasks in.
    :param n_task_completers: The number of task completers (processes that run the `_multiprocessing_complete_tasks`
                              function). A stop mark will be sent to the tasks queue for each task completer.
    """
    while True:
        # Get the batch:
        batch: List[dict] = batches_queue.get()
        if batch == _MULTIPROCESSING_STOP_MARK:
            break

        # Process the batch:
        batch_processor.process_batch(batch=batch)

        # Get the tasks:
        tasks = batch_processor.get_tasks()

        # Queue the tasks:
        for task in tasks:
            tasks_queue.put(task.to_tuple())

    # Mark the end of the batches:
    for _ in range(n_task_completers):
        tasks_queue.put(_MULTIPROCESSING_STOP_MARK)


def _multiprocessing_complete_tasks(tasks_queue: Queue, results_queue: Queue):
    """
    Complete the tasks in the given queue and put the results in the given results queue. The function will stop when
    the given tasks queue will receive the stop mark. It is aimed to be used with multiprocessing as a process.

    :param tasks_queue:   A queue to get the tasks from.
    :param results_queue: A queue to put the results in.
    """
    tasks_map = {
        BaseTask.__name__: BaseTask,
        SpeechDiarizationTask.__name__: SpeechDiarizationTask,
        SpeechDiarizationPerChannelTask.__name__: SpeechDiarizationPerChannelTask,
    }

    while True:
        # Get the task:
        task = tasks_queue.get()
        if task == _MULTIPROCESSING_STOP_MARK:
            break

        # Reconstruct the task:
        task_class, task_kwargs = task
        task = tasks_map[task_class](**task_kwargs)

        # Complete the task:
        task.do_task()
        results_queue.put((task.is_failed(), task.get_result()))

    # Mark the end of the tasks:
    results_queue.put(_MULTIPROCESSING_STOP_MARK)


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
                        data_path=Path(input_argument).absolute()
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

            # Save the output directory of this worker:
            output_directory = Path(output[0])

            # Send the output to the root rank (rank #0):
            output = comm.gather(output, root=0)

            # Join the data from all workers:
            if rank == 0:
                context.logger.info("Collecting data from workers to root worker.")

                # Check if there are different output directories:
                output_directories = set([Path(out_dir) for out_dir, _, _ in output])
                for r in range(1, size):
                    # True means the other workers should pass their files to the root worker (rank 0):
                    comm.send(len(output_directories) != 1, dest=r)

                # If there are different output directories, listen to the other workers:
                if len(output_directories) != 1:
                    # Collect the files from the other workers:
                    files = []
                    for r in range(1, size):
                        files.extend(comm.recv(source=r))
                    # Write the files to the root worker's output directory:
                    for file_name, file_content in files:
                        with open(output_directory / file_name, "w") as f:
                            f.write(file_content)

                # Concatenate the dataframes:
                dataframe = pd.concat(objs=[df for _, df, _ in output], axis=0)

                # Concatenate the errors dictionaries:
                errors_dictionary = reduce(
                    operator.ior, [err for _, _, err in output], {}
                )

                return str(output_directory), dataframe, errors_dictionary

            # Listen to rank 0 to see if there are different output directories and this rank need to send its files to
            # it:
            if comm.recv(source=0):
                files = []
                for file in os.listdir(output_directory):
                    with open(output_directory / file, "r") as f:
                        files.append((file, f.read()))
                comm.send(files, dest=0)
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
    # Input / Output kwargs:
    data_path: Union[str, Path, List[Union[str, Path]]],
    output_directory: str = None,
    # Model loading kwargs:
    model_name: str = "openai/whisper-tiny",
    device: str = None,
    use_flash_attention_2: bool = None,
    use_better_transformers: bool = None,
    # Generation kwargs:
    assistant_model: str = None,
    max_new_tokens: int = 128,
    chunk_length_s: int = 30,
    batch_size: int = 8,
    spoken_language: str = None,
    translate_to_english: bool = False,
    # Diarization kwargs:
    speech_diarization: Dict[str, List[Tuple[float, float, str]]] = None,
    speech_diarize_per_channel: int = None,
    speaker_labels: List[str] = None,
    # Other kwargs:
    use_multiprocessing: Union[bool, int] = False,
    verbose: bool = False,
):
    """
    Transcribe audio files into text files and collect additional data. The end result is a directory of transcribed
    text files and a dataframe containing the following columns:

    * audio_file - The audio file path.
    * transcription_file - The transcribed text file name in the output directory.

    The transcription is based on Huggingface's ASR pipeline -
    https://huggingface.co/transformers/main_classes/pipelines.html#transformers.AutomaticSpeechRecognitionPipeline and
    is tested with OpenAI's Whisper models - https://huggingface.co/openai.

    If one of the speaker diarization parameters are given (either `speech_diarization` or
    `speech_diarize_per_channel`), the transcription will be written in a conversation format, where each speaker will
    be written in a separate line::

        speaker_1: text
        speaker_2: text
        speaker_1: text
        ...

    :param data_path:                  A directory of audio files or a single file or a list of files to transcribe.
    :param output_directory:           Path to a directory to save all transcribed audio files. If not given, will save
                                       the transcribed files in a temporary directory.
    :param model_name:                 The model name to use. Should be a model from the OpenAI's Whisper models for
                                       best results (for example "tiny", "base", "large", etc.). See here for more
                                       information: https://huggingface.co/openai?search_models=whisper.
    :param device:                     The device to use for inference. If not given, will use GPU if available.
    :param use_flash_attention_2:      Whether to use the Flash Attention 2 implementation. It can be used only with
                                       one of the following GPUs: Nvidia H series and Nvidia A series. T4 support
                                       will be available soon.

                                       Note: If both `use_flash_attention_2` and
                                       `use_better_transformers` are `None`, the optimization will be chosen
                                       automatically according to the available resources.

    :param use_better_transformers:    Whether to use the Better Transformers library to further optimize the model.
                                       Should be used for all use cases that do not support flash attention 2.

                                       Note: If both `use_flash_attention_2` and `use_better_transformers` are
                                       `None`, the optimization will be chosen automatically according to the
                                       available resources.
    :param assistant_model:            The assistant model name to use for inference. Notice that the optimizations
                                       (flash attention 2 and better transformers) will be applied for the assistant as
                                       well. Should be a model from Huggingface's distil-whisper (see here for more
                                       information: https://github.com/huggingface/distil-whisper).

                                       Note: Currently an assistant model is only usable with batch size of 1.
    :param max_new_tokens:             The maximum number of new tokens to generate. This is used to limit the
                                       generation length. Default is 128 tokens.
    :param chunk_length_s:             The audio chunk to split the audio to (in seconds). Default is 30 seconds.
    :param batch_size:                 The batch size to use for inference. Default is 2.
    :param spoken_language:            Aim whisper to know what language is spoken. If None, it will try to detect
                                       it.
    :param translate_to_english:       Whether to translate the transcriptions to English.
    :param speech_diarization:         A speech diarization dictionary with the file names to transcribe as keys and
                                       their diarization as value. The diarization is a list of tuples:
                                       (start, end, speaker). An example
                                       for a diarization dictionary::

                                       {
                                           "audio_file_name": [
                                               {
                                                   "start": 0.0,
                                                   "end": 2.0,
                                                   "speaker": "Agent",
                                               },
                                               {
                                                   "start": 2.0,
                                                   "end": 4.0,
                                                   "speaker": "Client",
                                               },
                                               ...
                                           ],
                                           ...
                                       }

                                       Note: The diarization must be for the entire duration of the audio file (as long
                                       as Whisper is predicting words up until then.
    :param speech_diarize_per_channel: Perform speech diarization per channel. Each speaker is expected to belong to
                                       a separate channel in the audio. Notice: This will make the transcription
                                       slower as each channel wil be transcribed separatly. If a speech diarization
                                       is passed (via the `speech_diarization` parameter), this parameter is
                                       ignored.
    :param speaker_labels:             A list of speaker labels by channel order to use for writing the
                                       transcription with respect to per channel speech diarization. This won't be
                                       used together with a given speech diarization (via the `speech_diarization`
                                       parameter).
    :param use_multiprocessing:        Whether to use multiprocessing to transcribe the audio files. Can be either a
                                       boolean value or an integer. If `True`, will use the default amount of workers
                                       (3): 1 for transcription, 1 for batch processing and 1 for task completion (such
                                       as speech diarization and writing to files). To control the amount of tasks
                                       completion workers, an integer can be provided to specify the amount of workers.
                                       `False`, will use a single process. Default is `False`.
    :param verbose:                    Whether to print the progress of the transcription. Default is `False`.
    """
    global _LOGGER

    # Get the input audio files to transcribe:
    if verbose:
        _LOGGER.info("Collecting audio files.")
    audio_files = _get_audio_files(data_path=data_path)
    if verbose:
        _LOGGER.info(f"Collected {len(audio_files)} audio files.")

    # Get the output directory:
    if output_directory is None:
        if verbose:
            _LOGGER.info("No output directory given, using temporary directory.")
        output_directory = tempfile.mkdtemp()
    output_directory = Path(output_directory).absolute()
    output_directory.mkdir(exist_ok=True, parents=True)
    if verbose:
        _LOGGER.info(f"Transcriptions will be saved to: {output_directory}")

    # Initialize a batch processor according to user requirements (no speech diarization, given speech diarization,
    # speech diarization per channel):
    if speech_diarization:
        batch_processor = SpeechDiarizationBatchProcessor(
            audio_files=audio_files,
            output_directory=output_directory,
            speech_diarization=speech_diarization,
        )
    elif speech_diarize_per_channel:
        batch_processor = PerChannelSpeechDiarizationBatchProcessor(
            audio_files=audio_files,
            output_directory=output_directory,
            n_channels=speech_diarize_per_channel,
            speakers=speaker_labels,
        )
    else:
        batch_processor = BatchProcessor(
            audio_files=audio_files,
            output_directory=output_directory,
        )

    # Initialize the transcription pipeline:
    transcriber = Transcriber(
        device=device,
        use_flash_attention_2=use_flash_attention_2,
        use_better_transformers=use_better_transformers,
        assistant_model=assistant_model,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        return_timestamps=(
            "word"
            if speech_diarization is not None or speech_diarize_per_channel is not None
            else False
        ),
        per_channel_transcription=speech_diarize_per_channel or 0,
        spoken_language=spoken_language,
        translate_to_english=translate_to_english,
    )

    # Run the transcription:
    if use_multiprocessing:
        results = _parallel_run(
            n_workers=use_multiprocessing
            if isinstance(use_multiprocessing, int)
            else 1,
            audio_files=audio_files,
            batch_processor=batch_processor,
            transcriber=transcriber,
            verbose=verbose,
        )
    else:
        results = _run(
            audio_files=audio_files,
            batch_processor=batch_processor,
            transcriber=transcriber,
            verbose=verbose,
        )

    # Process the results:
    if verbose:
        _LOGGER.info("Summarizing the results.")
    successes = []
    errors = {}
    for is_error, result in results:
        if is_error:
            errors[result[0]] = result[1]
        else:
            successes.append(result)
    successes = pd.DataFrame(successes, columns=["audio_file", "transcription_file"])
    if verbose:
        _LOGGER.info(
            f"Done ({successes.shape[0]}/{len(audio_files)})\n"
            f"Transcriptions summary:\n"
            f"{successes.head()}"
        )

    return str(output_directory), successes, errors


def _get_audio_files(
    data_path: Union[Path, str, list],
) -> List[Path]:
    """
    Get the audio files to transcribe. If a path to a directory is given, all files in the directory will be collected.

    :param data_path: The data path to collect the audio files from.

    :returns: The audio files list.
    """
    # Check if given a list of paths:
    if isinstance(data_path, list):
        audio_files = []
        for path in data_path:
            audio_files.extend(_get_audio_files(data_path=path))
        return audio_files

    # Check if given a single string path to cast it to a `pathlib.Path`:
    if isinstance(data_path, str):
        data_path = Path(data_path).absolute()

    # Check if the path is of a directory or a file:
    if data_path.is_dir():
        # Get all files inside the directory:
        audio_files = list(data_path.glob("*.*"))
    elif data_path.is_file():
        audio_files = [data_path]
    else:
        raise ValueError(
            f"Unrecognized data path. The parameter `data_path` must be a valid path to either a directory path or a "
            f"file. Given: {str(data_path)} "
        )

    return audio_files


def _run(
    audio_files: List[Path],
    batch_processor: BatchProcessor,
    transcriber: Transcriber,
    verbose: bool,
) -> List[Tuple[bool, Tuple[str, str]]]:
    """
    Run the transcription without multiprocessing.

    :param audio_files:     The audio files to transcribe.
    :param batch_processor: The batch processor to use.
    :param transcriber:     The transcriber to use.
    :param verbose:         Verbosity.

    :returns: The collected results.
    """
    # Load the transcription pipeline:
    if verbose:
        _LOGGER.info(f"Loading the transcription pipeline.")
    transcriber.load()
    if verbose:
        _LOGGER.info("Transcription pipeline loaded.")

    # Transcribe the files:
    transcriber.transcribe(
        audio_files=audio_files,
        batch_processor=batch_processor,
        verbose=verbose,
    )

    # Return the results:
    return batch_processor.get_results()


def _parallel_run(
    n_workers: int,
    audio_files: List[Path],
    batch_processor: BatchProcessor,
    transcriber: Transcriber,
    verbose: bool,
):
    """
    Run the transcription with multiprocessing.

    :param n_workers:       The amount of workers to use as task completers.
    :param audio_files:     The audio files to transcribe.
    :param batch_processor: The batch processor to use.
    :param transcriber:     The transcriber to use.
    :param verbose:         Verbosity.

    :returns: The collected results.
    """
    # Initialize the multiprocessing queues:
    batches_queue = Queue()
    tasks_queue = Queue()
    results_queue = Queue()

    # Initialize the multiprocessing processes:
    batch_processing_process = Process(
        target=_multiprocessing_process_batches,
        kwargs={
            "batch_processor": batch_processor,
            "batches_queue": batches_queue,
            "tasks_queue": tasks_queue,
            "n_task_completers": n_workers,
        },
    )
    task_completion_processes = [
        Process(
            target=_multiprocessing_complete_tasks,
            kwargs={"tasks_queue": tasks_queue, "results_queue": results_queue},
        )
        for _ in range(n_workers)
    ]

    # Start the multiprocessing processes:
    batch_processing_process.start()
    for p in task_completion_processes:
        p.start()

    # Load the transcription pipeline:
    if verbose:
        _LOGGER.info(f"Loading the transcription pipeline.")
    transcriber.load()
    if verbose:
        _LOGGER.info("Transcription pipeline loaded.")

    # Transcribe the files:
    transcriber.transcribe(
        audio_files=audio_files, batches_queue=batches_queue, verbose=verbose
    )

    # Collect the results:
    results = []
    stop_marks_counter = 0
    while True:
        # Get a result from the queue:
        result: Tuple[bool, Tuple[str, str]] = results_queue.get()
        if result == _MULTIPROCESSING_STOP_MARK:
            stop_marks_counter += 1
            if stop_marks_counter == n_workers:
                break
        else:
            # Collect the result:
            results.append(result)

    # Wait for the processes to finish:
    results_queue.empty()
    batch_processing_process.join()
    for p in task_completion_processes:
        p.join()

    return results