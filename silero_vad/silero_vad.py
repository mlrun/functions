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
from multiprocessing import Process, Queue
from pathlib import Path
from types import FunctionType
from typing import Dict, List, Tuple, Type, Union

import torch
import torchaudio
from tqdm import tqdm


class BaseTask:
    """
    A base class for a task to complete after VAD.
    """

    def __init__(self, audio_file: Path):
        """
        Initialize the base task.

        :param audio_file: The audio file assigned to the task.
        """
        # Store the audio file:
        self._audio_file = audio_file

        # Prepare the result:
        self._result = None

    @property
    def audio_file(self) -> Path:
        """
        Get the audio file of the task.

        :returns: The audio file of the task.
        """
        return self._audio_file

    def do_task(
        self, speech_timestamps: Union[List[Dict[str, int]], List[List[Dict[str, int]]]]
    ):
        """
        Do the task on the given speech timestamps. The base task will simply save the speech timestamps as the result.

        :param speech_timestamps: The speech timestamps to do the task on as outputted from the VAD.
        """
        self._result = speech_timestamps

    def get_result(self) -> Tuple[str, list]:
        """
        Get the result of the task. A tuple of the audio file name and the result.

        :returns: The result of the task.
        """
        return self._audio_file.name, self._result

    def to_tuple(self) -> Tuple[str, dict]:
        """
        Convert the task to a tuple to reconstruct it later (used for multiprocessing to pass in queue).

        :returns: The converted task.
        """
        return self.__class__.__name__, {"audio_file": self._audio_file}


class SpeechDiarizationTask(BaseTask):
    """
    A speech diarization task. The task will diarize the VAD speech timestamps into speakers.
    """

    def __init__(self, audio_file: Path, speaker_labels: List[str]):
        """
        Initialize the speech diarization task.

        :param audio_file:     The audio file assigned to the task.
        :param speaker_labels: The speaker labels to use for the diarization. If not given, the speakers will be named
                               "speaker_0", "speaker_1", etc.
        """
        super().__init__(audio_file=audio_file)
        self._speaker_labels = speaker_labels

    def do_task(self, speech_timestamps: List[List[Dict[str, int]]]):
        """
        Do the task on the given speech timestamps. The task will diarize the VAD speech timestamps into speakers.

        :param speech_timestamps: The speech timestamps per channel to do the task on as outputted from the VAD.
        """
        # Get the speaker labels (set default if not given):
        speaker_labels = self._speaker_labels or [
            f"speaker_{i}" for i in range(len(speech_timestamps))
        ]

        # Diarize - organize the speech timestamps into a single list of speakers and sort it by start time:
        speech_diarization = [
            (speech_timestamp["start"], speech_timestamp["end"], speaker_label)
            for speaker_label, channel_speech_timestamps in zip(
                speaker_labels, speech_timestamps
            )
            for speech_timestamp in channel_speech_timestamps
        ]
        speech_diarization.sort()
        self._result = speech_diarization

    def to_tuple(self) -> Tuple[str, dict]:
        """
        Convert the task to a tuple to reconstruct it later (used for multiprocessing to pass in queue).

        :returns: The converted task.
        """
        task_class, task_kwargs = super().to_tuple()
        return task_class, {**task_kwargs, "speaker_labels": self._speaker_labels}


class TaskCreator:
    """
    A task creator to create different tasks to run after the VAD.
    """

    #: A map from task class name to task class to use in `from_tuple`:
    _MAP = {
        BaseTask.__name__: BaseTask,
        SpeechDiarizationTask.__name__: SpeechDiarizationTask,
    }

    def __init__(self, task_type: Type[BaseTask], task_kwargs: dict = None):
        """
        Initialize the task creator.
        :param task_type: The task type - a `BaseTask` subclass.
        :param task_kwargs: Additional keyword arguments to pass to the to be created tasks.
        """
        self._task_type = task_type
        self._task_kwargs = task_kwargs or {}

    def create_task(self, audio_file: Path) -> BaseTask:
        """
        Create a task with the given audio file.

        :param audio_file: The audio file to assign to the task.

        :returns: The created task.
        """
        return self._task_type(audio_file=audio_file, **self._task_kwargs)

    @classmethod
    def from_tuple(cls, task_tuple: Tuple[str, dict]) -> BaseTask:
        """
        Create a task from a tuple of the audio file name and the task kwargs.

        :param task_tuple: The task tuple to create the task from.

        :returns: The created task.
        """
        task_class, task_kwargs = task_tuple
        return cls._MAP[task_class](**task_kwargs)


class VoiceActivityDetector:
    """
    A voice activity detection wrapper for the silero VAD model - https://github.com/snakers4/silero-vad.
    """

    def __init__(
        self,
        # Model loading kwargs:
        use_onnx: bool = True,
        force_onnx_cpu: bool = True,
        # Detection kwargs:
        threshold: float = 0.5,
        sampling_rate: int = 16_000,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float("inf"),
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30,
        return_seconds: bool = False,
        per_channel: bool = False,
    ):
        """
        Initialize the voice activity detector.

        :param use_onnx:                Whether to use ONNX for inference. Default is True.
        :param force_onnx_cpu:          Whether to force ONNX to use CPU for inference. Default is True.
        :param threshold:               Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
                                        probabilities ABOVE this value are considered as SPEECH. It is better to tune
                                        this parameter for each dataset separately, but "lazy" 0.5 is pretty good for
                                        most datasets.
        :param sampling_rate:           Currently, silero VAD models support 8000 and 16000 sample rates.
        :param min_speech_duration_ms:  Final speech chunks shorter min_speech_duration_ms are thrown out.
        :param max_speech_duration_s:   Maximum duration of speech chunks in seconds. Chunks longer than
                                        `max_speech_duration_s` will be split at the timestamp of the last silence that
                                        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise,
                                        they will be split aggressively just before max_speech_duration_s.
        :param min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms before
                                        separating it.
        :param window_size_samples:     Audio chunks of window_size_samples size are fed to the silero VAD model.
                                        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000
                                        sample rate and 256, 512, 768 samples for 8000 sample rate. Values other than
                                        these may affect model performance!
        :param speech_pad_ms:           Final speech chunks are padded by speech_pad_ms each side.
        :param return_seconds:          Whether return timestamps in seconds. False means to return timestamps in
                                        samples (default - False).
        :param per_channel:             Whether to return timestamps per channel (default - False). This will run VAD
                                        on each channel separately and return a list of timestamps per channel.
        """
        # Store configurations:
        self._use_onnx = use_onnx
        self._force_onnx_cpu = force_onnx_cpu
        self._threshold = threshold
        self._sampling_rate = sampling_rate
        self._min_speech_duration_ms = min_speech_duration_ms
        self._max_speech_duration_s = max_speech_duration_s
        self._min_silence_duration_ms = min_silence_duration_ms
        self._window_size_samples = window_size_samples
        self._speech_pad_ms = speech_pad_ms
        self._return_seconds = return_seconds
        self._per_channel = per_channel

        # Prepare the model variables
        self._model: torch.Module = None
        self._get_speech_timestamps: FunctionType = None

    def load(self, force_reload: bool = True):
        """
        Load the VAD model.

        :param force_reload: Whether to force reload the model even if it was already loaded. Default is True.
        """
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=force_reload,
            onnx=self._use_onnx,
            force_onnx_cpu=self._force_onnx_cpu,
        )
        self._model = model
        (
            self._get_speech_timestamps,
            _,  # save_audio,
            _,  # read_audio,
            _,  # VADIterator,
            _,  # collect_chunks
        ) = utils

    def detect_voice(
        self,
        audio_file: Path,
    ) -> Union[List[Dict[str, int]], List[List[Dict[str, int]]]]:
        """
        Infer the audio through the VAD model and return the speech timestamps.

        :param audio_file: The audio file to infer.

        :returns: The speech timestamps in the audio. A list of timestamps where each timestamp is a dictionary with the
                 following keys:

                 * "start": The start sample index of the speech in the audio.
                 * "end":   The end sample index of the speech in the audio.

                 If `per_channel` is True, a list of timestamps per channel will be returned.
        """
        # Cast to a numpy array:
        audio = self._read_audio(audio_file)

        # Detect speech:
        if not self._per_channel:
            return self._get_speech_timestamps(
                audio,
                self._model,
                threshold=self._threshold,
                min_speech_duration_ms=self._min_speech_duration_ms,
                max_speech_duration_s=self._max_speech_duration_s,
                min_silence_duration_ms=self._min_silence_duration_ms,
                speech_pad_ms=self._speech_pad_ms,
                sampling_rate=self._sampling_rate,
                window_size_samples=self._window_size_samples,
                return_seconds=self._return_seconds,
            )

        # Per channel:
        speech_timestamps = []
        for channel in audio:
            speech_timestamps.append(
                self._get_speech_timestamps(
                    channel,
                    self._model,
                    threshold=self._threshold,
                    min_speech_duration_ms=self._min_speech_duration_ms,
                    max_speech_duration_s=self._max_speech_duration_s,
                    min_silence_duration_ms=self._min_silence_duration_ms,
                    speech_pad_ms=self._speech_pad_ms,
                    sampling_rate=self._sampling_rate,
                    window_size_samples=self._window_size_samples,
                    return_seconds=self._return_seconds,
                )
            )

        return speech_timestamps

    def _read_audio(
        self,
        path: Path,
    ) -> torch.Tensor:
        """
        Read the audio from the given path and return it as a tensor.

        :param path: The path to the audio file.

        :returns: The audio as a tensor.
        """
        # Read the audio:
        audio, sampling_rate = torchaudio.load(str(path))

        # Check if the audio is stereo and if so, convert it to mono (only if not per channel):
        if audio.size(0) > 1 and not self._per_channel:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample the audio if needed:
        if sampling_rate != self._sampling_rate:
            transform = torchaudio.transforms.Resample(
                orig_freq=sampling_rate, new_freq=self._sampling_rate
            )
            audio = transform(audio)

        # Return the audio (squeeze if not per channel):
        return audio if self._per_channel else audio.squeeze(0)


#: The value to send into multiprocessing queues to stop the process:
_MULTIPROCESSING_STOP_MARK = "STOP"


def _multiprocessing_complete_tasks(
    vad_init_kwargs: dict, tasks_queue: Queue, results_queue: Queue
):
    """
    Complete the tasks in the given queue and put the results in the given results queue. The function will stop when
    the given tasks queue will receive the stop mark. It is aimed to be used with multiprocessing as a process.

    :param vad_init_kwargs: The VAD initialization kwargs.
    :param tasks_queue:     A queue to get the tasks from.
    :param results_queue:   A queue to put the results in.
    """
    # Initialize and load the VAD:
    vad = VoiceActivityDetector(**vad_init_kwargs)
    vad.load(force_reload=False)

    # Start listening to the tasks queue:
    while True:
        # Get the task:
        task: Tuple[str, dict] = tasks_queue.get()
        if task == _MULTIPROCESSING_STOP_MARK:
            break
        try:
            # Create the task:
            task = TaskCreator.from_tuple(task_tuple=task)
            # Run the file through the VAD:
            speech_timestamps = vad.detect_voice(audio_file=task.audio_file)
            # Complete the task:
            task.do_task(speech_timestamps=speech_timestamps)
            # Build the result:
            result = (False, task.get_result())
        except Exception as exception:
            # Build the error:
            result = (True, (task.audio_file.name, str(exception)))
        # Collect the result / error:
        results_queue.put(result)

    # Mark the end of the tasks:
    results_queue.put(_MULTIPROCESSING_STOP_MARK)


# Get the global logger:
try:
    import mlrun

    _LOGGER = mlrun.get_or_create_ctx("silero_vad").logger
except ModuleNotFoundError:
    _LOGGER = logging.getLogger()


def detect_voice(
    # Input kwargs:
    data_path: Union[str, Path, List[Union[str, Path]]],
    # Model loading kwargs:
    use_onnx: bool = True,
    force_onnx_cpu: bool = True,
    # Detection kwargs:
    threshold: float = 0.5,
    sampling_rate: int = 16_000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 512,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    per_channel: bool = False,
    # Other kwargs:
    use_multiprocessing: int = 0,
    verbose: bool = False,
):
    """
    Perform voice activity detection on given audio files using the silero VAD model -
    https://github.com/snakers4/silero-vad. The end result is a dictionary with the file names as keys and their
    VAD timestamps dictionaries as value.

    For example::

        {
            "file_1.wav": [
                {"start": 0, "end": 16000},
                {"start": 16000, "end": 32000},
                {"start": 32000, "end": 48000},
                ...
            ],
            "file_2.wav": [
                {"start": 0, "end": 16000},
                {"start": 16000, "end": 32000},
                {"start": 32000, "end": 48000},
                ...
            ],
            ...
        }


    :param data_path:               The path to the audio files to diarize. Can be a path to a single file, a path to a
                                    directory or a list of paths to files.
    :param use_onnx:                Whether to use ONNX for inference. Default is True.
    :param force_onnx_cpu:          Whether to force ONNX to use CPU for inference. Default is True.
    :param threshold:               Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
                                    probabilities ABOVE this value are considered as SPEECH. It is better to tune
                                    this parameter for each dataset separately, but "lazy" 0.5 is pretty good for
                                    most datasets.
    :param sampling_rate:           Currently, silero VAD models support 8000 and 16000 sample rates.
    :param min_speech_duration_ms:  Final speech chunks shorter min_speech_duration_ms are thrown out.
    :param max_speech_duration_s:   Maximum duration of speech chunks in seconds. Chunks longer than
                                    `max_speech_duration_s` will be split at the timestamp of the last silence that
                                    lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will
                                    be split aggressively just before max_speech_duration_s.
    :param min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms before separating
                                    it.
    :param window_size_samples:     Audio chunks of window_size_samples size are fed to the silero VAD model.

                                    WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000
                                    sample rate and 256, 512, 768 samples for 8000 sample rate. Values other than
                                    these may affect model performance!
    :param speech_pad_ms:           Final speech chunks are padded by speech_pad_ms each side.
    :param return_seconds:          Whether return timestamps in seconds. False means to return timestamps in samples
                                    (default - False).
    :param per_channel:             Whether to return timestamps per channel (default - False). This will run VAD on
                                    each channel separately and return a list of timestamps per channel.
    :param use_multiprocessing:     The number of workers to use for multiprocessing. If 0, no multiprocessing will
                                    be used. Default is 0.
    :param verbose:                 Verbosity.
    """
    global _LOGGER

    # Get the input audio files to transcribe:
    if verbose:
        _LOGGER.info("Collecting audio files.")
    audio_files = _get_audio_files(data_path=data_path)
    if verbose:
        _LOGGER.info(f"Collected {len(audio_files)} audio files.")

    # Initialize the transcription pipeline:
    vad_init_kwargs = {
        "use_onnx": use_onnx,
        "force_onnx_cpu": force_onnx_cpu,
        "threshold": threshold,
        "sampling_rate": sampling_rate,
        "min_speech_duration_ms": min_speech_duration_ms,
        "max_speech_duration_s": max_speech_duration_s,
        "min_silence_duration_ms": min_silence_duration_ms,
        "window_size_samples": window_size_samples,
        "speech_pad_ms": speech_pad_ms,
        "return_seconds": return_seconds,
        "per_channel": per_channel,
    }

    # Create the task creator:
    task_creator = TaskCreator(task_type=BaseTask)

    # Run the transcription:
    if use_multiprocessing:
        results = _parallel_run(
            n_workers=use_multiprocessing,
            audio_files=audio_files,
            description="Detecting voice",
            vad_init_kwargs=vad_init_kwargs,
            task_creator=task_creator,
            verbose=verbose,
        )
    else:
        results = _run(
            audio_files=audio_files,
            description="Detecting voice",
            vad_init_kwargs=vad_init_kwargs,
            task_creator=task_creator,
            verbose=verbose,
        )

    # Process the results:
    return _process_results(results=results, verbose=verbose)


def diarize(
    # Input / Output kwargs:
    data_path: Union[str, Path, List[Union[str, Path]]],
    # Model loading kwargs:
    use_onnx: bool = True,
    force_onnx_cpu: bool = True,
    # Detection kwargs:
    threshold: float = 0.5,
    sampling_rate: int = 16_000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 512,
    speech_pad_ms: int = 30,
    # Diarization kwargs:
    speaker_labels: List[str] = None,
    # Other kwargs:
    use_multiprocessing: int = 0,
    verbose: bool = False,
):
    """
    Perform speech diarization on given audio files using the silero VAD model - https://github.com/snakers4/silero-vad.
    The speech diarization is performed per channel so that each channel in the audio belong to a different speaker. The
    end result is a dictionary with the file names as keys and their diarization as value. A diarization is a list
    of tuples: (start, end, speaker_label).

    For example::

        {
            "file_1.wav": [
                (0.0, 1.0, "speaker_0"),
                (1.0, 2.0, "speaker_1"),
                (2.0, 3.0, "speaker_0"),
                ...
            ],
            "file_2.wav": [
                (0.0, 1.0, "speaker_0"),
                (1.0, 2.0, "speaker_1"),
                (2.0, 3.0, "speaker_0"),
                ...
            ],
            ...
        }


    :param data_path:               The path to the audio files to diarize. Can be a path to a single file, a path to a
                                    directory or a list of paths to files.
    :param use_onnx:                Whether to use ONNX for inference. Default is True.
    :param force_onnx_cpu:          Whether to force ONNX to use CPU for inference. Default is True.
    :param threshold:               Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
                                    probabilities ABOVE this value are considered as SPEECH. It is better to tune
                                    this parameter for each dataset separately, but "lazy" 0.5 is pretty good for
                                    most datasets.
    :param sampling_rate:           Currently, silero VAD models support 8000 and 16000 sample rates.
    :param min_speech_duration_ms:  Final speech chunks shorter min_speech_duration_ms are thrown out.
    :param max_speech_duration_s:   Maximum duration of speech chunks in seconds. Chunks longer than
                                    `max_speech_duration_s` will be split at the timestamp of the last silence that
                                    lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will
                                    be split aggressively just before max_speech_duration_s.
    :param min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms before separating
                                    it.
    :param window_size_samples:     Audio chunks of window_size_samples size are fed to the silero VAD model.

                                    WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000
                                    sample rate and 256, 512, 768 samples for 8000 sample rate. Values other than
                                    these may affect model performance!
    :param speech_pad_ms:           Final speech chunks are padded by speech_pad_ms each side.
    :param speaker_labels:          The speaker labels to use for the diarization. If not given, the speakers will be
                                    named "speaker_0", "speaker_1", etc.
    :param use_multiprocessing:     The number of workers to use for multiprocessing. If 0, no multiprocessing will
                                    be used. Default is 0.
    :param verbose:                 Verbosity.
    """
    global _LOGGER

    # Get the input audio files to transcribe:
    if verbose:
        _LOGGER.info("Collecting audio files.")
    audio_files = _get_audio_files(data_path=data_path)
    if verbose:
        _LOGGER.info(f"Collected {len(audio_files)} audio files.")

    # Initialize the transcription pipeline:
    vad_init_kwargs = {
        "use_onnx": use_onnx,
        "force_onnx_cpu": force_onnx_cpu,
        "threshold": threshold,
        "sampling_rate": sampling_rate,
        "min_speech_duration_ms": min_speech_duration_ms,
        "max_speech_duration_s": max_speech_duration_s,
        "min_silence_duration_ms": min_silence_duration_ms,
        "window_size_samples": window_size_samples,
        "speech_pad_ms": speech_pad_ms,
        "return_seconds": True,
        "per_channel": True,
    }

    # Create the task creator:
    task_creator = TaskCreator(
        task_type=SpeechDiarizationTask, task_kwargs={"speaker_labels": speaker_labels}
    )

    # Run the transcription:
    if use_multiprocessing:
        results = _parallel_run(
            n_workers=use_multiprocessing,
            audio_files=audio_files,
            description="Diarizing",
            vad_init_kwargs=vad_init_kwargs,
            task_creator=task_creator,
            verbose=verbose,
        )
    else:
        results = _run(
            audio_files=audio_files,
            description="Diarizing",
            vad_init_kwargs=vad_init_kwargs,
            task_creator=task_creator,
            verbose=verbose,
        )

    # Process the results:
    return _process_results(results=results, verbose=verbose)


def _get_audio_files(
    data_path: Union[Path, str, list],
) -> List[Path]:
    """
    Get the audio files from the data path. If a path to a directory is given, all files in the directory will be
    collected.

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
    description: str,
    vad_init_kwargs: dict,
    task_creator: TaskCreator,
    verbose: bool,
) -> List[Tuple[bool, Tuple[str, list]]]:
    """
    Load a VAD and use it to complete the tasks that will be created on the provided files using the given task creator.

    :param audio_files:     The audio files to use.
    :param description:     The description to use for the progress bar.
    :param vad_init_kwargs: The VAD initialization keyword arguments.
    :param task_creator:    The task creator to use to create the tasks.
    :param verbose:         Verbosity.

    :returns: The collected results.
    """
    # Load the VAD:
    vad = VoiceActivityDetector(**vad_init_kwargs)
    if verbose:
        _LOGGER.info(f"Loading the VAD model.")
    vad.load()
    if verbose:
        _LOGGER.info("VAD model loaded.")

    # Run the VAD on the audio files and collect the results:
    results = []
    for audio_file in tqdm(
        audio_files,
        desc=description,
        unit="file",
        total=len(audio_files),
        disable=not verbose,
    ):
        try:
            # Create the task:
            task = task_creator.create_task(audio_file=audio_file)
            # Run the file through the VAD:
            speech_timestamps = vad.detect_voice(audio_file=audio_file)
            # Complete the task:
            task.do_task(speech_timestamps=speech_timestamps)
            # Collect the result:
            results.append((False, task.get_result()))
        except Exception as exception:
            # Collect the error:
            results.append((True, (audio_file.name, str(exception))))

    return results


def _parallel_run(
    n_workers: int,
    audio_files: List[Path],
    description: str,
    vad_init_kwargs: dict,
    task_creator: TaskCreator,
    verbose: bool,
) -> List[Tuple[bool, Tuple[str, list]]]:
    """
    Run multiple VAD workers with multiprocessing to complete the tasks that will be created on the provided files using
    the given task creator.

    :param n_workers:       The number of workers to use.
    :param audio_files:     The audio files to use.
    :param description:     The description to use for the progress bar.
    :param vad_init_kwargs: The VAD initialization keyword arguments.
    :param task_creator:    The task creator to use to create the tasks.
    :param verbose:         Verbosity.

    :returns: The collected results.
    """
    # Load the VAD (download once, and it will be loaded then per process later on):
    if verbose:
        _LOGGER.info(f"Loading the VAD model.")
    vad = VoiceActivityDetector(**vad_init_kwargs)
    vad.load()
    if verbose:
        _LOGGER.info("VAD model loaded.")

    # Check the number of workers:
    if n_workers > len(audio_files):
        _LOGGER.warning(
            f"The number of workers ({n_workers}) is larger than the number of audio files ({len(audio_files)}). "
            f"Setting the number of workers to {len(audio_files)}."
        )
        n_workers = len(audio_files)

    # Initialize the multiprocessing queues:
    tasks_queue = Queue()
    results_queue = Queue()

    # Initialize the multiprocessing processes:
    task_completion_processes = [
        Process(
            target=_multiprocessing_complete_tasks,
            kwargs={
                "vad_init_kwargs": vad_init_kwargs,
                "tasks_queue": tasks_queue,
                "results_queue": results_queue,
            },
        )
        for _ in range(n_workers)
    ]

    # Start the multiprocessing processes:
    for p in task_completion_processes:
        p.start()

    # Put the tasks in the queue:
    for audio_file in audio_files:
        tasks_queue.put(task_creator.create_task(audio_file=audio_file).to_tuple())

    # Put the stop marks in the queue:
    for _ in range(n_workers):
        tasks_queue.put(_MULTIPROCESSING_STOP_MARK)

    # Collect the results:
    results = []
    stop_marks_counter = 0
    with tqdm(
        desc=description,
        unit="file",
        total=len(audio_files),
        disable=not verbose,
    ) as progressbar:
        while True:
            # Get a result from the queue:
            result: Tuple[bool, Tuple[str, list]] = results_queue.get()
            if result == _MULTIPROCESSING_STOP_MARK:
                stop_marks_counter += 1
                if stop_marks_counter == n_workers:
                    break
            else:
                # Collect the result:
                results.append(result)
                progressbar.update(1)

    # Wait for the processes to finish:
    for p in task_completion_processes:
        p.join()

    return results


def _process_results(
    results: List[Tuple[bool, Tuple[str, list]]], verbose: bool
) -> Tuple[dict, dict]:
    """
    Process the results of the tasks.

    :param results: The results to process.
    :param verbose: Verbosity.

    :returns: The processed results as a tuple of successes and errors.
    """
    if verbose:
        _LOGGER.info("Summarizing the results.")
    successes = {}
    errors = {}
    for is_error, result in results:
        if is_error:
            errors[result[0]] = result[1]
        else:
            successes[result[0]] = result[1]
    if verbose:
        _LOGGER.info(f"Done ({len(successes)}/{len(successes) + len(errors)})\n")

    return successes, errors
