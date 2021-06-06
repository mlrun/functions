import subprocess
from pathlib import Path
from typing import Union, List, Set
import sys
import re
from pygit2 import Repository
import yaml
from jinja2 import Template
import shutil


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
        f"export PIP_NO_INPUT=1;pip install pipenv", stdout=sys.stdout, stderr=subprocess.PIPE, shell=True
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


def get_item_yaml_requirements(item_path: str):
    item_path = Path(item_path)
    if item_path.is_dir():
        item_path = item_path / "item.yaml"
    with open(item_path, "r") as f:
        item = yaml.full_load(f)
    requirements = item.get("spec", {}).get("requirements", [])
    requirements = requirements or []
    return requirements


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


def _set_mlrun_hub_url(repo_name = None, branch_name = None, function_name = None):
    repo_name =  re.search("\.com/.*?/", str(subprocess.run(['git', 'remote', '-v'], stdout=subprocess.PIPE).stdout)).group()[5:-1] if not repo_name else repo_name
    branch_name = Repository('.').head.shorthand if not branch_name else branch_name
    function_name = "" if not function_name else function_name # MUST ENTER FUNCTION NAME !!!!
    hub_url = f"https://raw.githubusercontent.com/{repo_name}/functions/{branch_name}/{function_name}/function.yaml"


def _delete_outputs(paths):
    for path in paths:
        if Path(path).is_dir():
            shutil.rmtree(path)