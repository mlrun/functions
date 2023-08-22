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
import tempfile

import mlrun
import pytest
import transformers

FRUIT = "apple"
COLOR = "red"


def mock_pipeline_call(*args, **kwargs):
    return [{"generated_text": args[1] + COLOR}]


def _make_data_dir_for_test():
    data_dir = tempfile.mkdtemp()
    content = "The apple color is red."
    with open(data_dir + "/test_data.txt", "w") as f:
        f.write(content)
    return data_dir


@pytest.mark.parametrize("input_path", ["./data", "./data/test-data.txt"])
def test_question_answering(monkeypatch, input_path):
    monkeypatch.setattr(transformers.Pipeline, "__call__", mock_pipeline_call)
    artifact_path = tempfile.mkdtemp()
    qa_function = mlrun.import_function("function.yaml")
    qa_run = qa_function.run(
        handler="answer_questions",
        params={
            "model": "distilgpt2",
            "input_path": input_path,
            "text_wrapper": (
                "Given the following sentence:\n" "-----\n" "{}\n" "-----"
            ),
            "questions": [
                "What is the color of the apple?",
            ],
            "questions_columns": [
                "color",
            ],
            "generation_config": {
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.9,
                "early_stopping": True,
                "max_new_tokens": 20,
            },
        },
        returns=[
            "question_answering_df: dataset",
            "question_answering_errors: result",
        ],
        local=True,
        artifact_path=artifact_path,
    )
    qa_df = mlrun.get_dataitem(
        f"{artifact_path}/question-answering-answer-questions/0/question_answering_df.parquet"
    ).as_df()
    assert qa_df["color"][0] == COLOR
    assert qa_run.outputs["question_answering_errors"] == {}


@pytest.mark.skipif(
    condition=(
        not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_API_BASE")
    ),
    reason="This test requires openai credentials",
)
def test_openai():
    input_path = "./data"
    artifact_path = tempfile.mkdtemp()
    qa_function = mlrun.import_function("function.yaml")
    qa_run = qa_function.run(
        handler="answer_questions",
        params={
            "model": "gpt-3.5-turbo",
            "input_path": input_path,
            "text_wrapper": (
                "Given the following sentence:\n" "-----\n" "{}\n" "-----"
            ),
            "questions_wrapper": "Answer the questions in an indexed list format"
            " where each answer is one word length:\n"
            "{}",
            "with_indexed_list_start": False,
            "questions": [
                "What is the mentioned color?",
                "what is the fruit name?",
            ],
            "questions_columns": [
                "color",
                "fruit",
            ],
            "generation_config": {
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.9,
                "early_stopping": True,
                "max_new_tokens": 20,
            },
            "llm_source": "open-ai",
        },
        returns=[
            "question_answering_df: dataset",
            "question_answering_errors: result",
        ],
        local=True,
        artifact_path=artifact_path,
    )

    qa_df = mlrun.get_dataitem(
        f"{artifact_path}/question-answering-answer-questions/0/question_answering_df.parquet"
    ).as_df()
    assert qa_df["color"][0].casefold() == COLOR
    assert qa_df["fruit"][0].casefold() == FRUIT
    assert qa_run.outputs["question_answering_errors"] == {}
