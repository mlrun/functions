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
from pathlib import Path
from typing import Optional, Union

import click
import semver
import yaml
from black import format_str, FileMode
from mlrun import code_to_function
from yaml import full_load

from cli.helpers import is_item_dir
from cli.path_iterator import PathIterator


@click.command()
@click.option(
    "-i", "--item-path", help="Path to item.yaml file or a directory containing one"
)
@click.option(
    "-o", "--output-path", default=None, help="Path to code_to_function output"
)
@click.option(
    "-c",
    "--code_output",
    is_flag=True,
    default=False,
    help="If spec.filename is a notebook, should a python file be created",
)
@click.option(
    "-fmt",
    "--format_code",
    is_flag=True,
    default=False,
    help="If -c/--code_output is enabled, and -fmt/--format is enabled, the code output will be formatted",
)
@click.option(
    "-b",
    "--bump_version",
    is_flag=True,
    default=False,
    help="If -b/--bump_version is enabled, increase the minor version in the item.yaml file",
)
def item_to_function_cli(
    item_path: str, output_path: Optional[str], code_output: bool, format_code: bool, bump_version: bool
):
    item_to_function(item_path, output_path, code_output, format_code, bump_version)


def item_to_function(
    item_path: str,
    output_path: Optional[str] = None,
    code_output: bool = False,
    format_code: bool = True,
    bump_version: bool = False,
):
    item_path = Path(item_path)
    if item_path.is_dir():
        item_path = item_path / "item.yaml"

    # That means we are in a specific item directory
    if item_path.exists():
        _output_path = output_path or item_path.parent / "function.yaml"
        create_function_yaml(item_path, _output_path, code_output, format_code, bump_version)
    # That means we need to search for items inside this direcotry
    else:
        for inner_dir in PathIterator(
            root=item_path.parent,
            rule=is_item_dir,
            as_path=True,
        ):
            try:
                _output_path = output_path or (inner_dir / "function.yaml")
                create_function_yaml(inner_dir, _output_path, code_output, format_code, bump_version)
            except Exception as e:
                print(e)
                click.echo(f"{inner_dir.name}: Failed to generate function.yaml")


def set_nested(parent, key, value):
    if value and isinstance(value, dict):
        for k, v in value.items():
            if not hasattr(parent, key):
                setattr(parent, key, value)
                return
            set_nested(getattr(parent, key), k, v)
        return
    if hasattr(parent, key) and isinstance(value, list):
        old_value = getattr(parent, key)
        setattr(parent, key, value + old_value)
    else:
        setattr(parent, key, value)


def _get_item_yaml(item_path: Path) -> dict:
    if item_path.is_dir():
        if (item_path / "item.yaml").exists():
            item_path = item_path / "item.yaml"
        else:
            raise FileNotFoundError(f"{item_path} does not contain a item.yaml file")
    elif not item_path.exists():
        raise FileNotFoundError(f"{item_path} not found")

    item_yaml = full_load(open(item_path, "r"))
    return item_path, item_yaml


def create_function_yaml(
    item_path: Union[str, Path],
    output_path: Optional[str] = None,
    code_output: bool = False,
    format_code: bool = True,
    bump_version: bool = False,
):
    item_path = Path(item_path)
    if bump_version:
        bump_function_yaml_version(item_path)

    item_path, item_yaml = _get_item_yaml(item_path)

    filename = item_yaml.get("spec", {}).get("filename")
    filename = filename or item_yaml.get("example")
    filename = item_path.parent / filename
    filename = filename.resolve()

    _code_output = ""
    if code_output and filename.suffix == ".ipynb":
        _code_output = Path(filename)
        _code_output = _code_output.parent / f"{_code_output.stem}.py"
        _code_output = _code_output.resolve()

    spec = item_yaml.get("spec", {})

    function_object = code_to_function(
        name=item_yaml["name"],
        filename=str(filename),
        handler=spec.get("handler"),
        kind=spec.get("kind"),
        code_output=str(_code_output),
        image=spec.get("image"),
        description=item_yaml.get("description", ""),
        requirements=spec.get("requirements"),
        categories=item_yaml.get("categories", []),
        labels=item_yaml.get("labels", {}),
        with_doc=True,
    )
    function_object.metadata.project = ""

    custom_fields = spec.get("customFields", {})
    for key, value in custom_fields.items():
        setattr(function_object.spec, key, value)

    # Extra spec with nesting support (dict inside dict):
    extra_spec = spec.get("extra_spec")
    if extra_spec:
        set_nested(function_object, "spec", extra_spec)

    env = spec.get("env", {})
    if env:
        setattr(function_object.spec, "env", [])
    for key, value in env.items():
        function_object.spec.env.append({"name": key, "value": value})
        setattr(function_object.spec, key, value)

    if output_path is None:
        return function_object

    output_path = Path(output_path)

    if output_path.is_dir():
        output_path = output_path / "function.yaml"

    if not output_path.parent.exists():
        output_path.mkdir()

    function_object.export(target=str(output_path.resolve()))

    if code_output and format_code:
        with open(_code_output, "r") as file:
            code = file.read()
        code = format_str(code, mode=FileMode())
        with open(_code_output, "w") as file:
            file.write(code)


def bump_function_yaml_version(item_path: Path):
    item_path, item_yaml = _get_item_yaml(item_path)
    item_ver = item_yaml.get("version", "0.0.0")
    new_ver = semver.Version.parse(item_ver).bump_minor()
    item_yaml["version"] = str(new_ver)
    with open(item_path, 'w') as file:
        yaml.safe_dump(item_yaml, file, default_flow_style=False)


if __name__ == "__main__":
    # item_to_function_cli()
    item_to_function(
        "/home/michaell/projects/functions/tf1_serving"
    )