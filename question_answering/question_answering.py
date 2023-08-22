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
from abc import ABC, abstractmethod
from typing import List, Tuple

import mlrun
import pandas as pd
import transformers
from langchain.chat_models import ChatOpenAI
from tqdm.auto import tqdm


def answer_questions(
    context: mlrun.MLClientCtx,
    input_path: str,
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
    llm_source: str = "local",
    with_indexed_list_start: bool = True,
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
    :param llm_source:              Indicates which model to use
    :param with_indexed_list_start: Whether to add a '1. ' in the end of the prompt.

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
        with_indexed_list_start=with_indexed_list_start,
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

    llm = _get_model(
        context=context,
        llm_source=llm_source,
        model=model,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs,
        generation_config=generation_config,
        answer_preamble=answer_preamble,
    )

    # Go over the audio files and infer through the model:
    if pathlib.Path(input_path).is_file():
        input_path = [input_path]
    else:
        text_files_directory = pathlib.Path(input_path).absolute()
        input_path = list(text_files_directory.rglob("*.*"))

    for i, text_file in enumerate(
        tqdm(input_path, desc="Generating answers", unit="file")
    ):
        try:
            # Read the text:
            with open(text_file, "r") as fp:
                text = fp.read()
            # For each text, try multiple times as the llm might experience zero answers:
            tryout = 0
            answers = ""
            for tryout in range(model_answering_tryouts):
                # Infer through the llm:
                answers = llm._generate(prompt_template.format(text))
                # Validate the answers:
                if len(answers) != len(questions):
                    continue
                answers = [
                    answer[3:] for answer in answers
                ]  # Without the questions index prefix 'i. '
                if all(len(answer) == 0 for answer in answers):
                    continue
                # Note in the dataframe:
                df.loc[i - len(errors)] = [
                    pathlib.Path(text_file).name,
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


def _get_model(
    context,
    llm_source,
    model,
    tokenizer,
    model_kwargs,
    generation_config,
    answer_preamble,
):
    if llm_source == LLMSources.open_ai:
        llm = OpenAIModel(context, model)
    elif llm_source == LLMSources.local:
        llm = HuggingFaceModel(
            context=context,
            model=model,
            tokenizer=tokenizer,
            model_kwargs=model_kwargs,
            generation_config=generation_config,
            answer_preamble=answer_preamble,
        )
    else:
        raise ValueError(
            f"llm source '{llm_source}' not supported. Please use one of LLMSource values"
        )

    return llm


class LLModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _generate(self, **kwargs):
        pass


class HuggingFaceModel(LLModel):
    def __init__(
        self,
        context: mlrun.MLClientCtx,
        model: str,
        tokenizer: str,
        model_kwargs: dict,
        generation_config: dict,
        answer_preamble: str,
    ):
        self.answer_preamble = answer_preamble
        # Load the generation config:
        context.logger.info("Loading generation configuration")
        self.generation_config = transformers.GenerationConfig(
            **(generation_config or {})
        )
        context.logger.info("Generation configuration loaded.")

        # Load the tokenizer:
        tokenizer = tokenizer or model
        context.logger.info(f"Loading tokenizer: {tokenizer}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        context.logger.info("Tokenizer loaded")

        # Load the model and tokenizer into a pipeline object:
        context.logger.info(f"Loading model '{model}' and tokenizer into a pipeline")
        self.model = transformers.pipeline(
            task="text-generation",
            model=model,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        context.logger.info("Model loaded, pipeline created")

    def _generate(self, text: str):
        sequences = self.model(
            text,
            generation_config=self.generation_config,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self._postprocess(sequences[0]["generated_text"])

    def _postprocess(self, answers: str):
        answers = answers.split(f"{self.answer_preamble}\n", 1)[1]
        return answers.split("\n")


class OpenAIModel(LLModel):
    def __init__(self, context, model):
        # Setting OpenAI secrets in environment:
        context.logger.info("Setting OpenAI credentials in environment")
        _set_secret("OPENAI_API_KEY")
        _set_secret("OPENAI_API_BASE")
        context.logger.info("OpenAI credentials are set")

        context.logger.info("Connecting to OpenAI")
        self.model = ChatOpenAI(model=model)
        context.logger.info("Connection to OpenAI succeeded")

    def _generate(self, text):
        answers = self.model.predict(text)
        return self._postprocess(answers)

    @staticmethod
    def _postprocess(answers: str):
        return answers.split("\n")


class LLMSources:
    local = "local"
    open_ai = "open-ai"


def _set_secret(key: str):
    secret = mlrun.get_secret_or_env(key=key)
    if not secret:
        raise ValueError(
            f"Missing secret: {key}, please pass as secret or set as environment variable"
        )
    os.environ[key] = secret


def _get_prompt_template(
    text_wrapper: str,
    questions_wrapper: str,
    questions: List[str],
    answer_preamble: str,
    with_indexed_list_start: bool,
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

    if with_indexed_list_start:
        # Addition for better answers in general.
        indexed_list_start = "1. "
    else:
        indexed_list_start = ""
    # Construct the template:
    return (
        f"{text_wrapper}\n"
        f"{questions_wrapper.format(questions)}\n"
        "\n"
        f"{answer_preamble}\n"
        f"{indexed_list_start}"
    ), answer_preamble
