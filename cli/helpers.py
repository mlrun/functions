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
import subprocess
from pathlib import Path
from typing import Union, List, Set, Dict
import sys
from glob import iglob
import yaml
from jinja2 import Template

PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def is_item_dir(path: Path) -> bool:
    return path.is_dir() and (path / "item.yaml").exists()


def is_function_dir(path: Path) -> bool:
    if path.is_file():
        return False
    # dir_name = path.name
    # ipynb_found = any((f.name.endswith(".ipynb") for f in path.iterdir()))
    # py_found = any((f.name.endswith(".py") for f in path.iterdir()))
    return any((f.name == "function.yaml" for f in path.iterdir()))


def render_jinja(
    template_path: Union[str, Path], output_path: Union[str, Path], data: dict
):
    with open(template_path, "r") as t:
        template_text = t.read()

    template = Template(template_text)
    rendered = template.render(**data)

    with open(output_path, "w+") as out_t:
        out_t.write(rendered)


def install_pipenv():
    print("Installing pipenv...")
    pipenv_install: subprocess.CompletedProcess = subprocess.run(
        f"export PIP_NO_INPUT=1;pip install pipenv",
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        shell=True,
    )
    exit_on_non_zero_return(pipenv_install)


def install_python(directory: Union[str, Path]):
    print(f"Installing python for {directory}...")
    python_install: subprocess.CompletedProcess = subprocess.run(
        f"pipenv --rm;pipenv --python 3.7",
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        cwd=directory,
        shell=True,
    )

    exit_on_non_zero_return(python_install)

    stderr = python_install.stderr.decode("utf8")
    stderr = stderr.split("\n")
    python_location = [l for l in stderr if "Virtualenv location: " in l]
    if python_location.count(python_location)>0:
        python_location = (
            python_location[0].split("Virtualenv location: ")[-1] + "bin/python"
        )
    else:
        python_location = None
    return python_location


def _run_subprocess(cmd: str, directory):
    completed_process: subprocess.CompletedProcess = subprocess.run(
        cmd,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        shell=True,
        cwd=directory,
    )
    exit_on_non_zero_return(completed_process)


def install_requirements(
    directory: str,
    requirements: Union[List[str], Set[str]],
):
    """
    Installing requirements from a requirements list/set and from a requirements.txt file if found in directory
    :param directory:       The relevant directory were the requirements are installed and collected
    :param requirements:    Requirement list/set with or without bounds
    """
    requirements_file = Path(directory) / 'requirements.txt'

    if not requirements and not requirements_file.exists():
        print(f"No requirements found for {directory}...")
        return

    if requirements_file.exists():
        print(f"Installing requirements from {requirements_file}...")
        _run_subprocess(
            f"pipenv install --skip-lock -r {requirements_file}", directory
        )

    if requirements:
        print(f"Installing requirements [{' '.join(requirements)}] for {directory}...")
        _run_subprocess(
            f"pipenv install --skip-lock {' '.join(requirements)}", directory
        )


def get_item_yaml_values(
    item_path: pathlib.Path, keys: Union[str, Set[str]]
) -> Dict[str, Set[str]]:
    """
    Getting value from item.yaml requested field.

    :param item_path:       The path to the item.yaml file or the parent dir of the item.yaml.
    :param keys:            The fields names that contains the required values to collect,
                            also looks for the fields inside `spec` inside dict.

    :returns:               Set with all the values inside key.
    """
    if isinstance(keys, str):
        keys = {keys}
    values_dict = {}

    for key in keys:
        values_set = set()
        item_path = Path(item_path)
        if item_path.is_dir():
            item_path = item_path / "item.yaml"
        with open(item_path, "r") as f:
            item = yaml.full_load(f)
        if key in item:
            values = item.get(key, "")
        elif "spec" in item and key in item["spec"]:
            values = item["spec"].get(key, "") or ""
        else:
            values = ""

        if values:
            if isinstance(values, list):
                values_set = set(values)
            else:
                values_set.add(values)
        values_dict[key] = values_set

    return values_dict


def get_mock_requirements(source_dir: Union[str, Path]) -> List[str]:
    """
    Getting all requirements from .py files inside all the subdirectories of the given source dir.
    Only the files with the same name as their parent directory are taken in consideration.
    The requirements are being collected from rows inside the files that starts with `from` or `import`
    and parsed only to the base package.

    :param source_dir: The directory that contains all the functions.

    :return: A list of all the requirements.
    """
    mock_reqs = set()

    if isinstance(source_dir, Path):
        source_dir = source_dir.__str__()

    # Iterating over all .py files in the subdirectories:
    for filename in iglob(f"{source_dir}/**/*.py"):
        file_path = Path(filename)
        if file_path.parent.name != file_path.stem:
            # Skipping test files
            continue
        # Getting all packages:
        with open(filename, 'r') as f:
            lines = list(filter(None, f.read().split("\n")))
            for line in lines:
                words = line.split(' ')
                words = [w for w in words if w]
                if words and (words[0] == 'from' or words[0] == 'import'):
                    mock_reqs.add(words[1].split('.')[0])

    return sorted(mock_reqs)


def exit_on_non_zero_return(completed_process: subprocess.CompletedProcess):
    if completed_process.returncode != 0:
        print_std(completed_process)
        exit(completed_process.returncode)


def print_std(subprocess_result):
    print()
    print("==================== stdout ====================")
    if subprocess_result.stdout != None:
        print(subprocess_result.stdout.decode("utf-8"))
    print("==================== stderr ====================")
    if subprocess_result.stderr != None:
        print(subprocess_result.stderr.decode("utf-8"))
    print()
