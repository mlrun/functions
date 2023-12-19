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
import mlrun
import transformers
import tempfile

APPLE_COLOR = "red"


def mock_pipeline_call(*args, **kwargs):
    return [[{"generated_text": "1. " + APPLE_COLOR}]]


def _make_data_dir_for_test():
    data_dir = tempfile.mkdtemp()
    content = "The apple color is red."
    with open(data_dir + "/test_data.txt", "w") as f:
        f.write(content)
    return data_dir


def test_question_answering(monkeypatch):
    monkeypatch.setattr(transformers.Pipeline, "__call__", mock_pipeline_call)
    input_path = "./data"
    artifact_path = tempfile.mkdtemp()
    project = mlrun.new_project("qa", context="./")
    fn = project.set_function("question_answering.py", "answer_questions", kind="job", image="mlrun/mlrun")
    qa_run = fn.run(
        handler="answer_questions",
        params={
            "model_name": "distilgpt2",
            "data_path": input_path,
            "text_wrapper": [(
                "Given the following sentence:\n"
                "-----\n"
                "{}\n"
                "-----"
            )],
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
        artifact_path=artifact_path
    )
    qa_df = mlrun.get_dataitem(
        qa_run.status.artifacts[0]["spec"]["target_path"]
    ).as_df()
    assert qa_df["color"][0] == APPLE_COLOR
    assert qa_run.outputs["question_answering_errors"] == {}
