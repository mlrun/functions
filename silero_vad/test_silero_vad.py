import os
import tempfile

import mlrun
import pytest


@pytest.fixture()
def setup_test():
    with tempfile.TemporaryDirectory() as artifact_path:
        project = mlrun.get_or_create_project(name="default", context=artifact_path)
        func = project.set_function(
            func=os.path.abspath("./function.yaml"),
            name="silero-vad",
            image="mlrun/mlrun",
        )
        yield func, artifact_path


def test_detect_voice(setup_test):
    silero_vad_function, artifact_path = setup_test
    run = silero_vad_function.run(
        handler="detect_voice",
        inputs={"data_path": "./assets"},
        returns=["vad_outputs: file", "errors: file"],
        artifact_path=artifact_path,
        local=True,
    )
    assert run.outputs["vad_outputs"]


def test_diarize(setup_test):
    silero_vad_function, artifact_path = setup_test
    run = silero_vad_function.run(
        handler="diarize",
        inputs={"data_path": "./assets"},
        params={
            "speakers_labels": ["Agent", "Client"],
        },
        returns=["speech_diarization: file", "errors: file"],
        artifact_path=artifact_path,
        local=True,
    )
    assert run.outputs["speech_diarization"]
