import os

import mlrun
import pytest


@pytest.mark.skipif("HUGGING_FACE_HUB_TOKEN" not in os.environ, reason="no token")
def test_speech_diarization():
    project = mlrun.new_project("diarization-test2")
    speech_diarization = project.set_function(
        func="./function.yaml", name="speech_diarization", image="mlrun/mlrun"
    )

    diarize_run = speech_diarization.run(
        handler="diarize",
        inputs={"data_path": os.path.join("assets", "test_data.wav")},
        params={
            "device": "cpu",
            "speakers_labels": ["Agent", "Client"],
            "separate_by_channels": True,
        },
        returns=["speech_diarization: file", "diarize_errors: file"],
        local=True,
    )
    assert diarize_run.outputs["speech_diarization"]
