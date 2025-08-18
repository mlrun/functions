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
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import transformers
from tqdm import tqdm

# Get the global logger:
_LOGGER = logging.getLogger()


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
                if isinstance(input_argument, (str, pathlib.Path)):
                    input_argument = _get_text_files(
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


@open_mpi_handler(worker_inputs=["data_path"], root_worker_inputs={"verbose": True})
def translate(
    data_path: Union[str, List[str], pathlib.Path],
    output_directory: str,
    model_name: str = None,
    source_language: str = None,
    target_language: str = None,
    device: str = None,
    model_kwargs: dict = None,
    batch_size: int = 1,
    translation_kwargs: dict = None,
    verbose: bool = False,
) -> Tuple[str, pd.DataFrame, dict]:
    """
    Translate text files using a transformer model from Huggingface's hub according to the source and target languages
    given (or using the directly provided model name). The end result is a directory of translated text files and a
    dataframe containing the following columns:

    * text_file - The text file path.
    * translation_file - The translation text file name in the output directory.

    :param data_path:          A directory of text files or a single file or a list of files to translate.
    :param output_directory:   Directory where the translated files will be saved.
    :param model_name:         The name of a model to load. If None, the model name is constructed using the source and
                               target languages parameters.
    :param source_language:    The source language code (e.g., 'en' for English).
    :param target_language:    The target language code (e.g., 'en' for English).
    :param model_kwargs:       Keyword arguments to pass regarding the loading of the model in HuggingFace's `pipeline`
                               function.
    :param device:             The device index for transformers. Default will prefer cuda if available.
    :param batch_size:         The number of batches to use in translation. The files are translated one by one, but the
                               sentences can be batched.
    :param translation_kwargs: Additional keyword arguments to pass to a `transformers.TranslationPipeline` when doing
                               the translation inference. Notice the batch size here is being added automatically.
    :param verbose:            Whether to present logs of a progress bar and errors. Default: True.

    :returns: A tuple of:

              * Path to the output directory.
              * A dataframe dataset of the translated file names.
              * A dictionary of errored files that were not translated.
    """
    global _LOGGER

    # Get the input text files to translate:
    if verbose:
        _LOGGER.info("Collecting text files.")
    if isinstance(data_path, str):
        data_path = pathlib.Path(data_path).absolute()
        text_files = _get_text_files(data_path=data_path)
    else:
        text_files = data_path
    if verbose:
        _LOGGER.info(f"Collected {len(text_files)} text files.")

    # Get the translation pipeline:
    if verbose:
        _LOGGER.info(f"Loading model - using device '{device}'.")
    translation_pipeline, model_name = _get_translation_pipeline(
        model_name=model_name,
        source_language=source_language,
        target_language=target_language,
        device=device,
        model_kwargs=model_kwargs,
        batch_size=batch_size if batch_size != 1 else None,
    )
    if verbose:
        _LOGGER.info(f"Model '{model_name}' was loaded successfully.")

    # Prepare the successes dataframe and errors dictionary to be returned:
    successes = []
    errors = {}

    # Create the output directory:
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Prepare the translation keyword arguments:
    translation_kwargs = translation_kwargs or {}

    # Go over the audio files and transcribe:
    for text_file in tqdm(
        text_files, desc="Translating", unit="file", disable=not verbose
    ):
        try:
            # Translate:
            translation = _translate(
                text_file=text_file,
                translation_pipeline=translation_pipeline,
                translation_kwargs=translation_kwargs,
            )
            # Write the transcription to file:
            translation_file = _save_to_file(
                translation=translation,
                file_name=text_file.stem,
                output_directory=output_directory,
            )
            # Note as a success in the list:
            successes.append(
                [
                    text_file.name,
                    translation_file.name,
                ]
            )
        except Exception as exception:
            # Note the exception as error in the dictionary:
            if verbose:
                _LOGGER.warning(f"Error in file: '{text_file.name}'")
            errors[str(text_file.name)] = str(exception)
            continue

    # Construct the translations dataframe:
    columns = [
        "text_file",
        "translation_file",
    ]
    successes = pd.DataFrame(
        successes,
        columns=columns,
    )

    # Print the head of the produced dataframe and return:
    if verbose:
        _LOGGER.info(
            f"Done ({successes.shape[0]}/{len(text_files)})\n"
            f"Translations summary:\n"
            f"{successes.head()}"
        )
    return str(output_directory), successes, errors


def _get_text_files(
    data_path: pathlib.Path,
) -> List[pathlib.Path]:
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


def _get_translation_pipeline(
    model_name: str = None,
    source_language: str = None,
    target_language: str = None,
    device: str = None,
    model_kwargs: dict = None,
    batch_size: int = None,
) -> Tuple[transformers.Pipeline, str]:
    # Construct the model name - if model name is provided (not None) then we take it, otherwise we check both source
    # and target were provided to construct the model name:
    if model_name is None and (source_language is None or target_language is None):
        raise ValueError(
            "No model name were given and missing source and / or target languages. In order to translate you must "
            "pass a `model_name` or both `source_language` and `target_language`."
        )
    elif model_name is None:
        model_name = f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"

    # Initialize the translation pipeline:
    try:
        translation_pipeline = transformers.pipeline(
            task="translation",
            model=model_name,
            tokenizer=model_name,
            device=device,
            model_kwargs=model_kwargs,
            batch_size=batch_size,
        )
    except OSError as load_exception:
        if (
            "is not a valid model identifier listed on 'https://huggingface.co/models'"
            in str(load_exception)
            and source_language
        ):
            raise ValueError(
                f"The model '{model_name}' is not a valid model identifier. "
                f"The parameters `source_language` and `target_language` are used to construct a Helsinki model for "
                f"text to text generation, but the model created from the given languages does not exist. "
                f"You may check language identifiers at "
                f"https://developers.google.com/admin-sdk/directory/v1/languages, and if the error was not fixed, one "
                f"or more language code might be with 3 letters and needs to be found online. "
                f"Remember, you can always choose a model directly from the Huggingface hub by using the `model_name` "
                f"parameter."
            ) from load_exception
        raise load_exception

    return translation_pipeline, model_name


def _translate(
    text_file: pathlib.Path,
    translation_pipeline: transformers.Pipeline,
    translation_kwargs: dict,
) -> str:
    # Read the text from file:
    with open(text_file, "r") as fp:
        text = fp.read()

    # Split to paragraphs and each paragraph to sentences:
    paragraphs = [paragraph.split(".") for paragraph in text.split("\n")]

    # Discover the newline indexes to restore the file to its structure post translation:
    newlines_indexes = []
    for paragraph in paragraphs[:-1]:
        if len(newlines_indexes) == 0:
            newlines_indexes.append(len(paragraph) - 1)
        else:
            newlines_indexes.append(newlines_indexes[-1] + len(paragraph))

    # Prepare the batches (each sentence from the paragraphs). Notice we add a dot not only to restore the sentence
    # structure but to ignore empty strings as it will ruin the translation:
    sentences = [f"{line}." for paragraph in paragraphs for line in paragraph]

    # Translate the sentences:
    translations = translation_pipeline(sentences, **translation_kwargs)

    # Restructure the full text from the sentences:
    translated_text = []
    newline_index = newlines_indexes.pop(0) if newlines_indexes else None
    for i, translation in enumerate(translations):
        # Get the translation:
        text = translation["translation_text"]
        # Validate if it was an empty sentence before:
        if text == ".":
            text = ""
        # Check if needed to insert a newline:
        if newline_index and newline_index == i:
            text += "\n"
            newline_index = newlines_indexes.pop(0) if newlines_indexes else None
        # Collect it:
        translated_text.append(text)
    translated_text = "".join(translated_text)

    return translated_text


def _save_to_file(
    translation: str, file_name: str, output_directory: pathlib.Path
) -> pathlib.Path:
    # Prepare the file full path (checking for no duplications):
    translation_file = output_directory / f"{file_name}.txt"
    i = 1
    while translation_file.exists():
        i += 1
        translation_file = output_directory / f"{file_name}_{i}.txt"

    # Make sure all directories are created:
    translation_file.parent.mkdir(exist_ok=True, parents=True)

    # Write to file:
    with open(translation_file, "w") as fp:
        fp.write(translation)

    return translation_file
