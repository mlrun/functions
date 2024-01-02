import tempfile
from pathlib import Path
import pytest
import mlrun


@pytest.mark.parametrize("audio_source,expected_num_audio_files", [
    ("data/test_data.wav", 1),
    ("data/test_data.mp3", 1),
    ("data", 2),
])
def test_reduce_noise(audio_source, expected_num_audio_files):
    # set up the project and function
    artifact_path = tempfile.TemporaryDirectory().name
    project = mlrun.new_project("noise-reduction")
    noise_reduction_function = project.set_function("function.yaml")

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
    )

    # assert that the function run completed successfully
    _assert_audio_files_exist(
        audio_files=Path(noise_reduction_run.outputs["return"]),
        expected_num_audio_files=expected_num_audio_files
    )


@pytest.mark.parametrize("audio_source,expected_num_audio_files", [
    ("data/test_data.wav", 1),
    ("data/test_data.mp3", 1),
    ("data", 2),
])
def test_reduce_noise_dfn(audio_source, expected_num_audio_files):
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
    )

    # assert that the function run completed successfully
    _assert_audio_files_exist(
        audio_files=Path(noise_reduction_run.outputs["return"]),
        expected_num_audio_files=expected_num_audio_files
    )


def _assert_audio_files_exist(audio_files, expected_num_audio_files):
    assert audio_files.exists(), "Audio files directory do not exist"
    assert len(list(audio_files.glob("*.*"))) == expected_num_audio_files, "Number of audio files is not as expected"
