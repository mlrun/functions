from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile


def reduce_noise_dfn(
    audio_source: str,
    target_directory: str,
    pad: bool = True,
    atten_lim_db: int = None,
    **kwargs
):
    """
    Reduce noise from audio files using DeepFilterNet.
    Notice that the saved files are in wav format, even if the original files are in other format.

    :param audio_source:       path to audio file or directory of audio files
    :param target_directory:      path to target directory to save cleaned audio files
    :param pad:             whether to pad the audio file with zeros before cleaning
    :param atten_lim_db:    maximum attenuation in dB
    :param kwargs:          additional arguments to pass to torchaudio.load(). For more information see:
                            https://pytorch.org/audio/stable/generated/torchaudio.load.html
    """
    # import required packages
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
    except ImportError as e:
        raise ImportError("Please install deepfilternet packages") from e

    # create target directory:
    target_directory = _create_target_directory(target_directory)

    # get audio files:
    audio_files = _get_audio_files(audio_source)

    # Load default model
    model, df_state, _ = init_df()

    for file in tqdm(audio_files, desc="Cleaning audio files"):
        # load data
        audio, _ = load_audio(
            file=str(file),
            sr=df_state.sr(),
            **kwargs
        )

        # perform noise reduction
        reduced_noise = enhance(
            model=model,
            df_state=df_state,
            audio=audio,
            pad=pad,
            atten_lim_db=atten_lim_db,
        )

        target_path = file
        if target_path.suffix != ".wav":
            old_suffix = target_path.suffix[1:]
            target_path = target_path.with_stem(target_path.stem + f"_{old_suffix}")
            target_path = target_path.with_suffix(".wav")

        # save file
        save_audio(
            file=target_path.name,
            audio=reduced_noise,
            sr=df_state.sr(),
            output_dir=str(target_directory),
        )

    return target_directory


def reduce_noise(
    audio_source: str,
    target_directory: str,
    sample_rate: int = 16000,
    duration: int = None,
    channel: int = None,
):
    """
    Reduce noise from audio file or directory containing audio files.
    The audio files must be in .wav format.
    The cleaned audio files will be saved in the target_directory.
    For information about the noise reduction algorithm see:
    https://github.com/timsainb/noisereduce
    Notice that the saved files are in wav format, even if the original files are in other format.

    :param audio_source:   path to audio file or directory containing audio files
    :param target_directory:  path to directory to save the cleaned audio files.
    :param sample_rate: Number of samples in one second in the audio file.
                        Pass None to keep the original sample rate.
    :param duration:    Duration of the audio file to clean in seconds.
                        Pass None to keep the original duration.
    :param channel:     Channel to clean. Pass the number of the channel to clean.
                        To clean all channels pass None.
    """
    # import required packages
    try:
        import librosa
        import noisereduce
    except ImportError as e:
        raise ImportError("Please install librosa and noisereduce packages") from e

    # create target directory:
    target_directory = _create_target_directory(target_directory)

    # get audio files:
    audio_files = _get_audio_files(audio_source)

    for file in tqdm(audio_files, desc="Cleaning audio files"):
        # load data
        data, sr = librosa.load(
            path=file,
            sr=sample_rate,
            mono=False,  # keep channels separate
            duration=duration,
        )
        sr = int(sr)

        # convert to int with scaling for 16-bit integer
        data *= 32767 / np.max(np.abs(data))  # re-scaling
        data = data.astype(np.int16)  # change data type

        # select channel
        data_to_reduce = data[channel] if channel is not None else data

        # perform noise reduction
        reduced_noise = noisereduce.reduce_noise(y=data_to_reduce, sr=sr)

        # add channel back after noise reduction
        if channel is not None:
            # putting the channel back in the data
            data[channel] = reduced_noise
            # updating the data to save
            reduced_noise = data

        # Transpose if necessary, required for writing to file
        if len(reduced_noise) > 1:
            reduced_noise = reduced_noise.T

        # Modify suffix to .wav
        target_path = target_directory / file.name
        if target_path.suffix != ".wav":
            old_suffix = target_path.suffix[1:]
            target_path = target_path.with_stem(target_path.stem + f"_{old_suffix}")
            target_path = target_path.with_suffix(".wav")

        # save file
        wavfile.write(
            filename=target_path,
            rate=sr,
            data=reduced_noise,
        )

    return target_directory


def _create_target_directory(target_directory: str):
    target_directory = Path(target_directory)
    if not target_directory.exists():
        target_directory.mkdir(parents=True, exist_ok=True)
    return target_directory


def _get_audio_files(audio_source: str):
    audio_source = Path(audio_source)
    audio_files = []
    if audio_source.is_dir():
        audio_files = list(audio_source.glob("*.*"))
    elif audio_source.is_file():
        audio_files.append(audio_source)
    else:
        raise ValueError(f"audio_source must be a file or a directory, got {audio_source}")
    return audio_files
