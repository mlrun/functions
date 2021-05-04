import subprocess
from pathlib import Path
from typing import List

import click
import yaml

from common.helpers import is_item_dir
from common.path_iterator import PathIterator


@click.command()
@click.option("-r", "--root-directory", default=".", help="Path to root directory")
@click.option("-s", "--suite", help="Type of suite to run [py/ipynb/examples]")
def test_suite(root_directory: str, suite: str):
    if not suite:
        click.echo("-s/--suite is required")
        exit(1)

    if suite == "py":
        test_py(root_directory, clean=True)
    elif suite == "ipynb":
        test_ipynb(root_directory)
    elif suite == "examples":
        test_example(root_directory)
    else:
        click.echo(f"Suite {suite} is unsupported")
        exit(1)


def test_py(root_dir=".", clean=False):
    install_pipenv()

    for directory in PathIterator(root=root_dir, rule=is_item_dir):
        # install_python(directory)
        item_requirements = get_item_yaml_requirements(directory)
        install_requirements(directory, ["pytest"] + item_requirements)

        print(f"Running tests for {directory}...")
        run_tests: subprocess.CompletedProcess = subprocess.run(
            ["pipenv", "run", "python", "-m", "pytest"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=directory,
        )

        print_std(run_tests)

        if clean:
            clean_pipenv(directory)

        exit(run_tests.returncode)


def test_ipynb(root_dir=".", clean=False):
    install_pipenv()

    for directory in PathIterator(root=root_dir, rule=is_item_dir):
        notebooks = [n for n in PathIterator(root=directory, rule=is_test_notebook)]
        if not notebooks:
            continue

        # install_python(directory)
        item_requirements = get_item_yaml_requirements(directory)
        install_requirements(directory, ["papermill", "jupyter"] + item_requirements)

        for notebook in notebooks:
            print(f"Running tests for {notebook}...")
            run_papermill: subprocess.CompletedProcess = subprocess.run(
                ["pipenv", "run", "papermill", notebook, "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=directory,
            )
            if run_papermill.returncode != 0:
                print_std(run_papermill)
                exit(run_papermill.returncode)

        if clean:
            clean_pipenv(directory)

        exit(0)


def test_example(root_dir="."):
    install_pipenv()

    for directory in PathIterator(root=root_dir, rule=is_item_dir):
        notebooks = [n for n in PathIterator(root=directory, rule=is_test_notebook)]
        if not notebooks:
            continue

        # install_python(directory)
        item_requirements = get_item_yaml_requirements(directory)
        install_requirements(directory, ["papermill", "jupyter"] + item_requirements)

        # for notebook in notebooks:
        #     print(f"Running tests for {notebook}...")
        #     run_papermill: subprocess.CompletedProcess = subprocess.run(
        #         ["pipenv", "run", "papermill", notebook, "-"],
        #         stdout=subprocess.PIPE,
        #         stderr=subprocess.PIPE,
        #         cwd=directory,
        #     )
        #     if run_papermill.returncode != 0:
        #         print_std(run_papermill)
        #         exit(run_papermill.returncode)
        #     exit(0)


def clean(root_dir="."):
    for directory in PathIterator(root=root_dir, rule=is_item_dir):
        clean_pipenv(directory)


def is_test_notebook(path: Path) -> bool:
    return (
        path.is_file() and path.name.startswith("test") and path.name.endswith(".ipynb")
    )


def is_example_notebook(path: Path) -> bool:
    return (
        path.is_file()
        and not path.name.startswith("test")
        and path.name.endswith(".ipynb")
    )


def exit_on_non_zero_return(completed_process: subprocess.CompletedProcess):
    if completed_process.returncode != 0:
        print_std(completed_process)
        exit(completed_process.returncode)


def print_std(subprocess_result):
    print()
    print("==================== stdout ====================")
    print(subprocess_result.stdout.decode("utf-8"))
    print("==================== stderr ====================")
    print(subprocess_result.stderr.decode("utf-8"))
    print()


def clean_pipenv(directory: str):
    pip_file = Path(directory) / "Pipfile"
    pip_lock = Path(directory) / "Pipfile.lock"

    if pip_file.exists():
        pip_file.unlink()
    if pip_lock.exists():
        pip_lock.unlink()


def install_python(directory: str):
    print(f"Installing python for {directory}...")
    python_install: subprocess.CompletedProcess = subprocess.run(
        ["pipenv", "--python", "3.7"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=directory,
    )
    exit_on_non_zero_return(python_install)


def install_pipenv():
    print("Installing pipenv...")
    pipenv_install: subprocess.CompletedProcess = subprocess.run(
        ["pip", "install", "pipenv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    exit_on_non_zero_return(pipenv_install)


def install_requirements(directory: str, requirements: List[str]):
    if not requirements:
        print(f"No requirements found for {directory}...")
        return

    print(f"Installing requirements [{' '.join(requirements)}] for {directory}...")
    requirements_install: subprocess.CompletedProcess = subprocess.run(
        ["pipenv", "install", "--skip-lock"] + requirements,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=directory,
    )

    exit_on_non_zero_return(requirements_install)


def get_item_yaml_requirements(directory: str):
    with open(f"{directory}/item.yaml", "r") as f:
        item = yaml.full_load(f)
    return item.get("spec", {}).get("requirements", [])


if __name__ == "__main__":
    test_suite()
