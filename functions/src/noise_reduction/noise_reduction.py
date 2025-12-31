import logging
from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Queue
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm

#: The value to send into multiprocessing queues to stop the process:
_MULTIPROCESSING_STOP_MARK = "STOP"

# Get the global logger:
try:
    import mlrun

    _LOGGER = mlrun.get_or_create_ctx("noise_reduce").logger
except ModuleNotFoundError:
    _LOGGER = logging.getLogger()


class ReduceNoiseBase(metaclass=ABCMeta):
    """
    Base class for noise reduction.
    This class is aimed to be inherited by specific noise reduction algorithms.
    You must implement the following methods:
    - clean_audio:  The method to clean the audio, where the noise reduction algorithm is implemented.
    - save_audio:   The method to save the audio to a file.
    - load_audio:   The method to load the audio from a file.

    After implementing the above methods, you can use the reduce_noise method to reduce noise from audio files.
    """

    def __init__(
        self,
        target_directory: Path,
        verbose: bool = True,
        silence_threshold: float = None,
    ):
        self.target_directory = Path(target_directory)
        self.verbose = verbose
        self.silence_threshold = silence_threshold

    def reduce_noise(self, audio_file: Path) -> tuple[bool, tuple[str, str]]:
        """
        Reduce noise from the given audio file.

        :param audio_file:  The audio file to reduce noise from.

        :returns: A tuple of:
         - a boolean indicating whether an error occurred
         - a tuple of:
            - audio file name
            - target path in case of success / error message in case of failure.
        """
        try:
            if self.verbose:
                _LOGGER.info(f"Reducing noise from {audio_file.name}.")

            # Load audio data:
            audio = self.load_audio(file=str(audio_file))

            # Perform noise reduction:
            reduced_noise = self.clean_audio(data=audio)

            # Remove silence from the audio if necessary:
            reduced_noise = self.remove_silence(audio=reduced_noise)

            # Prepare target path:
            target_path = self.update_to_wav_suffix(audio_file=audio_file)

            # Save file:
            self.save_audio(
                audio=reduced_noise,
                target_path=target_path,
            )

            if self.verbose:
                _LOGGER.info(f"Saved cleaned audio file to {target_path}.")

            return False, (audio_file.name, str(target_path))
        except Exception as exception:
            if self.verbose:
                _LOGGER.error(f"Failed to reduce noise from {audio_file.name}.")
                _LOGGER.error(f"Error: {exception}")
            # Collect the error:
            return True, (audio_file.name, str(exception))

    @abstractmethod
    def clean_audio(self, data) -> np.ndarray | torch.Tensor:
        """
        Clean the audio from noise. Here you should implement the noise reduction algorithm.

        :param data:    The audio data to clean.

        :returns: The cleaned audio.
        """
        pass

    @abstractmethod
    def save_audio(self, audio: np.ndarray, target_path: Path):
        """
        Save the audio to a file.

        :param audio:       The audio to save.
        :param target_path: The target path to save the audio to.
        """
        pass

    @abstractmethod
    def load_audio(self, file: str) -> tuple[np.ndarray | torch.Tensor, int]:
        """
        Load the audio from a file.

        :param file:    The file to load the audio from.

        :returns: A tuple of:
            - the audio data
            - the sample rate
        """
        pass

    def update_to_wav_suffix(self, audio_file: Path):
        target_path = self.target_directory / audio_file.name
        if target_path.suffix != ".wav":
            old_suffix = target_path.suffix[1:]
            target_path = target_path.with_stem(target_path.stem + f"_{old_suffix}")
            return target_path.with_suffix(".wav")
        else:
            return target_path

    def remove_silence(
        self,
        audio: np.ndarray,
    ):
        """
        Remove silence sections from the audio.

        :param audio:   The audio to remove silence from.

        :returns: The audio without silence.
        """
        if self.silence_threshold is None:
            return audio

        # Get the indices of the non-silent frames:
        non_silent_indices = librosa.effects.split(
            y=audio,
            top_db=self.silence_threshold,
            frame_length=2048,
            hop_length=256,
        )

        # Get the non-silent audio:
        non_silent_audio = np.concatenate(
            [audio[:, start:end] for start, end in non_silent_indices], axis=1
        )

        return non_silent_audio


class ReduceNoise(ReduceNoiseBase):
    def __init__(
        self,
        target_directory: Path,
        verbose: bool = True,
        silence_threshold: float = None,
        sample_rate: int = 16000,
        duration: int = None,
        channel: int = None,
    ):
        super().__init__(target_directory, verbose, silence_threshold)
        self.sample_rate = sample_rate
        self.duration = duration
        self.channel = channel

    def save_audio(self, audio: np.ndarray, target_path: Path):
        # If the audio has more than one channel, transpose it in order to save it:
        if len(audio) > 1:
            audio = audio.T

        wavfile.write(
            filename=target_path,
            rate=self.sample_rate,
            data=audio,
        )

    def load_audio(self, file: str) -> np.ndarray:
        data, sr = librosa.load(
            path=file,
            sr=self.sample_rate,
            mono=False,  # keep channels separate
            duration=self.duration,
        )
        # set sample rate:
        self.sample_rate = int(sr)

        # convert to int with scaling for 16-bit integer
        data *= 32767 / np.max(np.abs(data))  # re-scaling
        data = data.astype(np.int16)  # change data type

        # select channel
        data_to_reduce = data[self.channel] if self.channel is not None else data
        return data_to_reduce

    def clean_audio(self, data: np.ndarray) -> np.ndarray:
        try:
            import noisereduce
        except ImportError as e:
            raise ImportError("Please install noisereduce package") from e

        reduced_noise = noisereduce.reduce_noise(y=data, sr=self.sample_rate)

        # add channel back after noise reduction
        if self.channel is not None:
            # putting the channel back in the data
            data[self.channel] = reduced_noise
            # updating the data to save
            reduced_noise = data

        return reduced_noise


class DFN(ReduceNoiseBase):
    def __init__(
        self,
        target_directory: Path,
        verbose: bool = True,
        silence_threshold: float = None,
        pad: bool = True,
        atten_lim_db: int = None,
        **kwargs,
    ):
        super().__init__(target_directory, verbose, silence_threshold)
        self.pad = pad
        self.atten_lim_db = atten_lim_db
        self.kwargs = kwargs

        # import required packages
        try:
            from df.enhance import init_df
        except ImportError as e:
            raise ImportError("Please install deepfilternet packages") from e

        if self.verbose:
            _LOGGER.info("Loading DeepFilterNet2 model.")

        # Load the model:
        model, df_state, _ = init_df()
        self.model = model
        self.df_state = df_state
        self.sample_rate = self.df_state.sr()

    def save_audio(self, audio: np.ndarray, target_path: Path):
        try:
            from df.enhance import save_audio
        except ImportError as e:
            raise ImportError("Please install deepfilternet package") from e
        save_audio(
            file=target_path.name,
            audio=audio,
            sr=self.sample_rate,
            output_dir=str(self.target_directory),
        )

    def load_audio(self, file: str) -> torch.Tensor:
        try:
            from df.enhance import load_audio
        except ImportError as e:
            raise ImportError("Please install deepfilternet package") from e
        audio, _ = load_audio(file=file, sr=self.sample_rate, **self.kwargs)
        return audio

    def clean_audio(self, data: torch.Tensor) -> torch.Tensor:
        try:
            from df.enhance import enhance
        except ImportError as e:
            raise ImportError("Please install deepfilternet package") from e
        return enhance(
            model=self.model,
            df_state=self.df_state,
            audio=data,
            pad=self.pad,
            atten_lim_db=self.atten_lim_db,
        )


def _multiprocessing_complete_tasks(
    noise_reduce_type: type[ReduceNoiseBase],
    noise_reduce_arguments: dict,
    tasks_queue: Queue,
    results_queue: Queue,
):
    """
    Complete the tasks in the given queue and put the results in the given results queue. The function will stop when
    the given tasks queue will receive the stop mark. It is aimed to be used with multiprocessing as a process.

    :param noise_reduce_type:       The noise reduce type to use.
    :param noise_reduce_arguments:  The noisereduce initialization kwargs.
    :param tasks_queue:             A queue to get the tasks from.
    :param results_queue:           A queue to put the results in.
    """
    # Initialize the reduce noise object
    noise_reducer = noise_reduce_type(**noise_reduce_arguments)

    # Start listening to the tasks queue:
    while True:
        # Get the audio_file:
        audio_file = tasks_queue.get()
        if audio_file == _MULTIPROCESSING_STOP_MARK:
            break
        audio_file = Path(audio_file)
        # Apply noise reduction and collect the result:
        results_queue.put(noise_reducer.reduce_noise(audio_file=audio_file))

    # Mark the end of the tasks:
    results_queue.put(_MULTIPROCESSING_STOP_MARK)


def reduce_noise_dfn(
    audio_source: str,
    target_directory: str,
    pad: bool = True,
    atten_lim_db: int = None,
    silence_threshold: float = None,
    use_multiprocessing: int = 0,
    verbose: bool = True,
    **kwargs,
):
    """
    Reduce noise from audio files using DeepFilterNet.
    For more information about the noise reduction algorithm see:
    https://github.com/Rikorose/DeepFilterNet
    Notice that the saved files are in wav format, even if the original files are in other format.

    :param audio_source:        path to audio file or directory of audio files
    :param target_directory:    path to target directory to save cleaned audio files
    :param pad:                 whether to pad the audio file with zeros before cleaning
    :param atten_lim_db:        maximum attenuation in dB
    :param silence_threshold:   the threshold to remove silence from the audio, in dB. If None, no silence removal is
                                performed.
    :param use_multiprocessing: Number of processes to use for cleaning the audio files.
                                If 0, no multiprocessing is used.
    :param verbose:             verbosity level. If True, display progress bar and logs.
    :param kwargs:              additional arguments to pass to torchaudio.load(). For more information see:
                                https://pytorch.org/audio/stable/generated/torchaudio.load.html
    """
    if verbose:
        _LOGGER.info("Reducing noise from audio files.")

    # create target directory:
    target_directory = _create_target_directory(target_directory)

    # get audio files:
    audio_files = _get_audio_files(audio_source)

    noise_reduce_arguments = {
        "target_directory": target_directory,
        "pad": pad,
        "atten_lim_db": atten_lim_db,
        "silence_threshold": silence_threshold,
        **kwargs,
    }

    if use_multiprocessing:
        results = _parallel_run(
            noise_reduce_type=DFN,
            noise_reduce_arguments=noise_reduce_arguments,
            n_workers=use_multiprocessing,
            audio_files=audio_files,
            description="Noise-reduction",
            verbose=verbose,
        )
    else:
        results = _run(
            noise_reduce_type=DFN,
            noise_reduce_arguments=noise_reduce_arguments,
            audio_files=audio_files,
            description="Noise-reduction",
            verbose=verbose,
        )

    return _process_results(results, verbose)


def reduce_noise(
    audio_source: str,
    target_directory: str,
    sample_rate: int = 16000,
    duration: int = None,
    channel: int = None,
    silence_threshold: float = None,
    use_multiprocessing: int = 0,
    verbose: bool = True,
):
    """
    Reduce noise from audio file or directory containing audio files.
    The audio files must be in .wav format.
    The cleaned audio files will be saved in the target_directory.
    For information about the noise reduction algorithm see:
    https://github.com/timsainb/noisereduce
    Notice that the saved files are in wav format, even if the original files are in other format.

    :param audio_source:        path to audio file or directory containing audio files
    :param target_directory:    path to directory to save the cleaned audio files.
    :param sample_rate:         Number of samples in one second in the audio file.
                                Pass `None` to keep the original sample rate.
    :param duration:            Duration of the audio file to clean in seconds.
                                Pass `None` to keep the original duration.
    :param channel:             Channel to clean. Pass the number of the channel to clean.
                                To clean all channels pass None.
    :param silence_threshold:   The threshold to remove silence from the audio, in dB.
                                If None, no silence removal is performed.
    :param use_multiprocessing: Number of processes to use for cleaning the audio files.
                                If 0, no multiprocessing is used.
    :param verbose:             Verbosity level. If True, display progress bar.
    """
    if verbose:
        _LOGGER.info("Reducing noise from audio files.")

    # create target directory:
    target_directory = _create_target_directory(target_directory)

    # get audio files:
    audio_files = _get_audio_files(audio_source)

    # Create the reduce noise object:
    noise_reduce_arguments = {
        "target_directory": target_directory,
        "sample_rate": sample_rate,
        "duration": duration,
        "channel": channel,
        "silence_threshold": silence_threshold,
    }

    if use_multiprocessing:
        results = _parallel_run(
            noise_reduce_type=ReduceNoise,
            noise_reduce_arguments=noise_reduce_arguments,
            n_workers=use_multiprocessing,
            audio_files=audio_files,
            description="Noise-reduction",
            verbose=verbose,
        )
    else:
        results = _run(
            noise_reduce_type=ReduceNoise,
            noise_reduce_arguments=noise_reduce_arguments,
            audio_files=audio_files,
            description="Noise-reduction",
            verbose=verbose,
        )

    return _process_results(results, verbose)


def _create_target_directory(target_directory: str) -> str:
    target_directory = Path(target_directory)
    if not target_directory.exists():
        target_directory.mkdir(parents=True, exist_ok=True)
    return str(target_directory)


def _get_audio_files(audio_source: str):
    audio_source = Path(audio_source)
    audio_files = []
    if audio_source.is_dir():
        audio_files = list(audio_source.glob("*.*"))
    elif audio_source.is_file():
        audio_files.append(audio_source)
    else:
        raise ValueError(
            f"audio_source must be a file or a directory, got {audio_source}"
        )
    return audio_files


def _parallel_run(
    noise_reduce_type: type[ReduceNoiseBase],
    noise_reduce_arguments: dict,
    n_workers: int,
    audio_files: list[Path],
    description: str,
    verbose: bool,
) -> list[tuple[bool, tuple[str, str]]]:
    """
    Run multiple noise reduce workers with multiprocessing to complete the tasks that will be created on the provided
    files using the given task creator.

    :param noise_reduce_type:   The noise reduce type to use.
    :param n_workers:           The number of workers to use.
    :param audio_files:         The audio files to use.
    :param description:         The description to use for the progress bar.
    :param verbose:             Verbosity.

    :returns: The collected results.
    """
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
                "noise_reduce_type": noise_reduce_type,
                "noise_reduce_arguments": noise_reduce_arguments,
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
        # tasks_queue.put(task_creator.create_task(audio_file=audio_file).to_tuple())
        tasks_queue.put(audio_file)

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
            result: tuple[bool, tuple[str, str]] = results_queue.get()
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


def _run(
    noise_reduce_type: type[ReduceNoiseBase],
    noise_reduce_arguments: dict,
    audio_files: list[Path],
    description: str,
    verbose: bool,
) -> list[tuple[bool, tuple[str, str]]]:
    """
    Run the noise reduce algorithm on the given audio files and collect the results.

    :param noise_reduce_type:       The noise reduce type to use.
    :param noise_reduce_arguments:  The noisereduce initialization kwargs.
    :param audio_files:             The audio files to use.
    :param description:             The description to use for the progress bar.
    :param verbose:                 Verbosity.

    :returns: The collected results.
    """
    # Create the reduce noise object:
    noise_reducer = noise_reduce_type(**noise_reduce_arguments)

    # Run the noise reduce algorithm on the audio files and collect the results:
    results = []
    for audio_file in tqdm(
        audio_files,
        desc=description,
        unit="file",
        total=len(audio_files),
        disable=not verbose,
    ):
        results.append(noise_reducer.reduce_noise(audio_file=audio_file))

    return results


def _process_results(
    results: list[tuple[bool, tuple[str, str]]], verbose: bool
) -> tuple[dict, dict]:
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
