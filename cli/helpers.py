import pathlib
import subprocess
from pathlib import Path
from typing import Union, List, Set, Iterable, Dict
import sys
import re
from glob import iglob
import itertools as it
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
    python_location = (
        python_location[0].split("Virtualenv location: ")[-1] + "bin/python"
    )
    return python_location


def install_requirements(directory: str, requirements: Union[List[str], Set[str]]):
    if not requirements:
        print(f"No requirements found for {directory}...")
        return

    print(f"Installing requirements [{' '.join(requirements)}] for {directory}...")
    requirements_install: subprocess.CompletedProcess = subprocess.run(
        f"pipenv install --skip-lock {' '.join(requirements)}",
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        shell=True,
        cwd=directory,
    )

    exit_on_non_zero_return(requirements_install)


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


def remove_version_constraints(requirements: Iterable) -> Set[str]:
    """
    Remove version constraints from requirements.
    For example:
        pytorch==1.2.3 -> pytorch
    Also ignores mlrun[] requirements.

    :param requirements:    An iterable that contains all the requirements as strings.

    :returns:                A set with all the parsed requirements.
    """
    version_chars = ["==", "~=", "<=", ">=", "<", ">", "!="]
    mlrun_pkgs_regex = re.compile(r"^mlrun\[.+]$")
    parsed_requirements = set()
    # Parsing the requirements by removing version constraints:
    for requirement in requirements:
        # For the case of mlrun[]
        if mlrun_pkgs_regex.search(requirement):
            continue
        else:
            # Looking for version constraint case:
            for version_char in version_chars:
                if version_char in requirement:
                    # Found version constraint and dropping the constraint:
                    requirement = requirement.split(version_char)[0]
                    break
        parsed_requirements.add(requirement)

    return parsed_requirements


def get_requirements_from_txt(requirements_path: str):
    """
    Collecting all requirements from requirements.txt.

    :param requirements_path:   The path to the requirements.txt or the parent dir of this file.
    """
    requirements = set()
    requirements_path = Path(requirements_path)
    if requirements_path.is_dir():
        requirements_path = requirements_path / "requirements.txt"
    if not requirements_path.exists():
        return set()
    with open(requirements_path, "r") as f:
        # removing empty lines:
        reqs = set(filter(None, f.read().split("\n")))

    for req in reqs:
        requirements.update({req, get_base_pkg(req)})
    return sorted(requirements)


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
