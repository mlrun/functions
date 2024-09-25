import tempfile

import mlrun
import pytest


@pytest.mark.parametrize(
    "audio_source",
    [
        "data/test_data.wav",
        "data/test_data.mp3",
        "data",
    ],
)
def test_reduce_noise(audio_source):
    # set up the project and function
    artifact_path = tempfile.TemporaryDirectory().name
    project = mlrun.new_project("noise-reduction")
    noise_reduction_function = project.set_function(
        func="function.yaml",
        name="reduce_noise",
        kind="job",
        image="mlrun/mlrun",
    )

    # run the function
    noise_reduction_run = noise_reduction_function.run(
        handler="reduce_noise",
        inputs={"audio_source": audio_source},
        params={
            "target_directory": artifact_path + "/data",
            "sample_rate": None,
        },
        local=True,
        artifact_path=artifact_path,
        returns=["successes: file", "errors: file"],
    )

    assert noise_reduction_run.outputs["successes"]


@pytest.mark.parametrize(
    "audio_source",
    [
        "data/test_data.wav",
        "data/test_data.mp3",
        "data",
    ],
)
def test_reduce_noise_dfn(audio_source):
    # set up the project and function
    artifact_path = tempfile.TemporaryDirectory().name
    project = mlrun.new_project("noise-reduction")
    noise_reduction_function = project.set_function(
        func="function.yaml",
        name="reduce_noise",
        kind="job",
        image="mlrun/mlrun",
    )

    # run the function
    noise_reduction_run = noise_reduction_function.run(
        handler="reduce_noise_dfn",
        inputs={"audio_source": audio_source},
        params={
            "target_directory": artifact_path + "/data",
            "atten_lim_db": 50,
        },
        local=True,
        artifact_path=artifact_path,
        returns=["successes: file", "errors: file"],
    )

    # assert that the function run completed successfully
    assert noise_reduction_run.outputs["successes"]
