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
from datetime import datetime
from pathlib import Path
from typing import Union

import click
import yaml

from functions.cli.utils.helpers import is_function_dir
from functions.cli.utils.path_iterator import PathIterator


@click.command()
@click.option(
    "-p",
    "-path",
    help="Path to one of: specific function.yaml, directory containing function.yaml or a root directory to search function.yamls in",
)
def function_to_item_cli(path: str):
    function_to_item(path)


def function_to_item(path: str):
    path = Path(path)

    if not path.exists():
        click.echo(f"{path} not found")
        exit(1)

    # If this is a directory, its either project root or a specific directory in the project
    if path.is_dir():
        function_path = path / "function.yaml"
        # If its a function directory
        if function_path.exists():
            item_path = path / "item.yaml"
            item = function_yaml_to_item(function_path)
            with open(item_path, "w") as f:
                yaml.dump(item, f)
        # Otherwise its a root directory to be iterated
        else:
            function_iterator = PathIterator(
                root=path, rule=is_function_dir, as_path=True
            )
            for function in function_iterator:
                function_path = function / "function.yaml"
                item_path = function / "item.yaml"
                item = function_yaml_to_item(function_path)
                with open(item_path, "w") as f:
                    yaml.dump(item, f)
    # Otherwise its a path to function.yaml
    else:
        path_dir = path.parent
        item_path = path_dir / "item.yaml"
        item = function_yaml_to_item(path)
        with open(item_path, "w") as f:
            yaml.dump(item, f)
    exit(0)


def function_yaml_to_item(function_path: Union[str, Path]) -> dict:

    function_path = Path(function_path)
    function_yaml = yaml.full_load(open(function_path))

    metadata = function_yaml.get("metadata", {})
    spec = function_yaml.get("spec", {})

    item = {
        "apiVersion": "v1",
        "categories": metadata.get("categories") or [],
        "description": spec.get("description") or "",
        "doc": "",
        "example": get_ipynb_file(function_path.parent),
        "generationDate": datetime.utcnow().strftime("%Y-%m-%d:%H-%M"),
        "icon": "",
        "labels": metadata.get("labels") or {},
        "maintainers": [],
        "mlrunVersion": "",
        "name": metadata.get("name") or "",
        "platformVersion": "",
        "spec": {
            "filename": get_py_file(function_path.parent),
            "handler": get_handler(function_yaml),
            "image": get_image(function_yaml),
            "kind": function_yaml.get("kind") or "",
            "requirements": get_requirements(function_yaml),
        },
        "url": "",
        "version": metadata.get("tag") or "0.0.1",
        "marketplaceType": "",
    }

    return item


def get_ipynb_file(path: Path) -> str:
    default = path / f"{path.name}.ipynb"

    if default.exists():
        return f"{path.name}.ipynb"

    ipynbs = list(filter(lambda d: d.suffix == ".ipynb", path.iterdir()))
    ipynbs = list(filter(lambda d: not d.name.startswith("test"), ipynbs))

    if len(ipynbs) > 1:
        click.echo(f"{path.name}: Notebook not found")
        return ""
    elif len(ipynbs) == 1:
        return ipynbs[0].name

    click.echo(f"{path.name}: Notebook not found")
    return ""


def get_py_file(path: Path) -> str:
    default_py_file = path / "function.py"

    if default_py_file.exists():
        return "function.py"

    py_file = filter(lambda d: d.suffix == ".py", path.iterdir())
    py_file = list(filter(lambda d: not d.name.startswith("test"), py_file))

    if len(py_file) > 1:
        click.echo(f"{path.name}: Python file not found")
        return ""
    elif len(py_file) == 1:
        return py_file[0].name

    click.echo(f"{path.name}: Python file not found")
    return ""


def get_handler(function_yaml: dict) -> str:
    spec = function_yaml["spec"]
    handler = spec.get("default_handler", "handler")
    return handler


def get_image(function_yaml: dict):
    spec = function_yaml.get("spec", {})
    spec_image = spec.get("image")

    build = spec.get("build", {})
    build_image = build.get("base_image")

    base_spec = spec.get("base_spec", {}).get("spec", {})
    base_spec_image = base_spec.get("build", {}).get("baseImage")

    return spec_image or build_image or base_spec_image or ""


def get_requirements(function_yaml: dict):
    spec = function_yaml.get("spec", {})
    base_spec = spec.get("base_spec", {}).get("spec", {})

    spec_commands = spec.get("build", {}).get("commands", [])
    base_spec_commands = base_spec.get("build", {}).get("commands", [])

    commands = set()

    for command in spec_commands:
        commands.add(command)

    for command in base_spec_commands:
        commands.add(command)

    requirements = []
    for command in commands:
        if "uninstall" in command:
            click.echo(f"{function_yaml['metadata']['name']}: Unsupported requirements")
            return []
        if "python -m pip install " in command:
            command = command.split("python -m pip install ")[-1]
        elif "pip install " in command:
            command = command.split("pip install ")[-1]
        else:
            click.echo(f"{function_yaml['metadata']['name']}: Unsupported requirements")
            return []

        allowed_chars = {".", "_", "-", "="}

        if " " in command:
            sub_commands = command.split(" ")
            for sub_command in sub_commands:
                for char in sub_command:
                    if not char.isalnum() and char not in allowed_chars:
                        click.echo(
                            f"{function_yaml['metadata']['name']}: Unsupported requirements"
                        )
                        return []
                requirements.append(sub_command)
        else:
            requirements.append(command)
    return requirements


if __name__ == "__main__":
    function_to_item_cli()
