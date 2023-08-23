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
from typing import List, Tuple

import mlrun
import pandas as pd
import transformers
from tqdm.auto import tqdm


def _get_prompt_template(
    text_wrapper: str,
    questions_wrapper: str,
    questions: List[str],
    answer_preamble: str,
) -> Tuple[str, str]:
    # Validate and build the text wrapper:
    text_wrapper = text_wrapper or (
        "Given the following text:\n" "-----\n" "{}\n" "-----"
    )
    if text_wrapper.count("{}") != 1:
        raise ValueError(
            "The `text_wrapper` must include one placeholder '{}' for the text."
        )

    # Validate and build the question wrapper:
    questions_wrapper = questions_wrapper or "Answer the questions:\n" "{}"
    if questions_wrapper.count("{}") != 1:
        raise ValueError(
            "The `questions_wrapper` must include one placeholder '{}' for the text."
        )

    # Validate and parse the questions:
    if len(questions) == 0:
        raise ValueError("Please include at least one question.")
    questions = "\n".join(
        [f"{i}. {question}" for i, question in enumerate(questions, 1)]
    )

    # Validate and build the answer wrapper:
    answer_preamble = answer_preamble or "Answer:"
    if answer_preamble.count("{}") != 0:
        raise ValueError("The `answer_preamble` must not have a placeholder '{}'.")

    # Construct the template:
    return (
        f"{text_wrapper}\n"
        f"{questions_wrapper.format(questions)}\n"
        "\n"
        f"{answer_preamble}\n"
        "1. "  # Addition for better answers in general.
    ), answer_preamble


def answer_questions(
    context: mlrun.MLClientCtx,
    input_path: Union[str.pathlib.Path],
    model: str,
    questions: List[str],
    tokenizer: str = None,
    model_kwargs: dict = None,
    text_wrapper: str = "",
    questions_wrapper: str = "",
    answer_preamble: str = "",
    generation_config: dict = None,
    questions_columns: List[str] = None,
    model_answering_tryouts: int = 2,
) -> Tuple[pd.DataFrame, dict]:
    """
    Answer questions with context to the given text files by a pretrained LLM model.

    :param context:                 MLRun context.
    :param input_path:              A path to a directory of text files or a path to a text file to ask questions about.
    :param model:                   The pre-trained model to use for asking questions.
    :param questions:               The questions to ask.
    :param tokenizer:               The pre-trained tokenizer to use. Defaulted to the model given.
    :param model_kwargs:            Keyword arguments to pass regarding the loading of the model in HuggingFace's
                                    `pipeline` function.
    :param text_wrapper:            A wrapper for the text part. Will be added at the start of the prompt. Must have a
                                    placeholder ('{}') for the questions.
    :param questions_wrapper:       A wrapper for the questions received. Will be added after the text placeholder in
                                    the prompt template. Must have a placeholder ('{}') for the questions.
    :param answer_preamble:         A prefix for the answer part. Will be added at the end of the prompt template.
    :param generation_config:       HuggingFace's `GenerationConfig` keyword arguments to pass to the `generate` method.
    :param questions_columns:       Columns to use for the dataframe returned.
    :param model_answering_tryouts: Amount of inferring to do per text before raising an error due to missing or empty
                                    answers.

    :returns: A tuple of:

              * A dataframe dataset of the questions answers.
              * A dictionary of errored files that were not inferred or were not answered properly.
    """
    # Get the prompt template:
    context.logger.info("Creating prompt template")
    prompt_template, answer_preamble = _get_prompt_template(
        text_wrapper=text_wrapper,
        questions_wrapper=questions_wrapper,
        questions=questions,
        answer_preamble=answer_preamble,
    )
    context.logger.info(f"Prompt template created:\n\n{prompt_template}\n")

    # Get the questions columns:
    questions_columns = questions_columns or [
        f"q{i}" for i in range(1, len(questions) + 1)
    ]
    if len(questions_columns) != len(questions):
        raise ValueError(
            f"The provided questions columns length ({len(questions_columns)}) "
            f"does not match the questions amount ({len(questions)})"
        )

    # Prepare the dataframe and errors to be returned:
    df = pd.DataFrame(columns=["text_file", *questions_columns])
    errors = {}

    # Load the tokenizer:
    tokenizer = tokenizer or model
    context.logger.info(f"Loading tokenizer: {tokenizer}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
    context.logger.info("Tokenizer loaded")

    # Load the generation config:
    context.logger.info("Loading generation configuration")
    generation_config = transformers.GenerationConfig(**(generation_config or {}))
    context.logger.info("Generation configuration loaded.")

    # Load the model and tokenizer into a pipeline object:
    context.logger.info(f"Loading model '{model}' and tokenizer into a pipeline")
    pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )
    context.logger.info("Model loaded, pipeline created")

    # Go over the audio files and infer through the model:
    if pathlib.Path(input_path).is_file():
        input_path = [input_path]
    else:
        text_files_directory = pathlib.Path(input_path).absolute()
        input_path = list(text_files_directory.rglob("*.*"))
    for i, text_file in enumerate(
        tqdm(
            input_path,
            desc="Generating answers",
            unit="file",
        )
    ):
        try:
            # Read the text:
            with open(text_file, "r") as fp:
                text = fp.read()
            # For each text, try multiple times as the llm might experience zero answers:
            tryout = 0
            for tryout in range(model_answering_tryouts):
                # Infer through the llm:
                sequences = pipeline(
                    prompt_template.format(text),
                    generation_config=generation_config,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                )
                # Validate the answers:
                answers = sequences[0]["generated_text"].split(
                    f"{answer_preamble}\n", 1
                )[1]
                answers = answers.split("\n")
                if len(answers) != len(questions):
                    continue
                answers = [
                    answer[3:] for answer in answers
                ]  # Without the questions index prefix 'i. '
                if all(len(answer) == 0 for answer in answers):
                    continue
                # Note in the dataframe:
                df.loc[i - len(errors)] = [
                    str(text_file.relative_to(text_files_directory)),
                    *answers,
                ]
                break
            if tryout == model_answering_tryouts:
                raise Exception(
                    f"The LLM did not answer correctly - one or more answers are missing: {answers}"
                )
        except Exception as exception:
            # Collect the exception:
            context.logger.error(f"Error in file: '{text_file}'")
            errors[str(text_file)] = str(exception)

    return df, errors
