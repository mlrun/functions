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
import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import click
import yaml
from bs4 import BeautifulSoup
from sphinx.cmd.build import main as sphinx_build_cmd
from sphinx.ext.apidoc import main as sphinx_apidoc_cmd

from cli.helpers import (PROJECT_ROOT, get_item_yaml_values,
                         get_mock_requirements, is_item_dir, render_jinja)
from cli.marketplace.changelog import ChangeLog
from cli.path_iterator import PathIterator

_verbose = False

# For preparing the assets section in catalog for each function
# The tuple values represents (<location in item.yaml>, <relative path value>)
ASSETS = {
    "example": ("example", "src/{}"),
    "source": ("spec.filename", "src/{}"),
    "function": "src/function.yaml",
    "docs": "static/documentation.html",
}


@click.command()
@click.option("-s", "--source-dir", help="Path to the source directory")
@click.option(
    "-sn",
    "--source-name",
    default="",
    help="Name of source, if not provided, name of source directory will be used instead",
)
@click.option("-m", "--marketplace-dir", help="Path to marketplace directory")
@click.option(
    "-T", "--temp-dir", default="/tmp", help="Path to intermediate build directory"
)
@click.option("-c", "--channel", default="master", help="Name of build channel")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="When this flag is set, the process will output extra information",
)
@click.option(
    "-f",
    "--force-update",
    "force_update_items",
    is_flag=True,
    default=False,
    help="When this flag is set, item pages will be created even if the item did not changed",
)
def build_marketplace_cli(
    source_dir: str,
    source_name: str,
    marketplace_dir: str,
    temp_dir: str,
    channel: str,
    verbose: bool,
    force_update_items: bool,
):
    build_marketplace(
        source_dir,
        marketplace_dir,
        source_name,
        temp_dir,
        channel,
        verbose,
        force_update_items,
    )


def build_marketplace(
    source_dir: str,
    marketplace_dir: str,
    source_name: Optional[str] = None,
    temp_dir: str = "/tmp",
    channel: str = "development",
    verbose: bool = False,
    force_update_items: bool = False,
):
    """Main entry point to marketplace building

    :param source_dir: Path to the source directory to build the marketplace from
    :param marketplace_dir: Path to marketplace directory
    :param source_name: Name of source, if not provided, name of source directory will be used instead
    :param temp_dir: Path to intermediate directory, used to build marketplace resources,
    if not provided '/tmp/<random_uuid>' will be used
    :param channel: The name of the marketplace channel to write to
    :param verbose: When True, additional debug information will be written to stdout
    :param force_update_items: If True, items will be updated unrelated if they are not changed.
                                The purpose of this flag is to fix existed broken pages (e.g. broken links)
    """
    global _verbose
    _verbose = verbose

    # The root of the temporary project
    root_base = Path(temp_dir) / uuid.uuid4().hex
    temp_root = root_base / "functions"
    temp_docs = root_base / "docs"

    click.echo(f"Temporary working directory: {root_base}")

    # The source directory of the marketplace
    source_dir = Path(source_dir).resolve()
    # The target directory of the marketplace
    marketplace_root = Path(marketplace_dir).resolve()
    marketplace_dir = marketplace_root / (source_name or source_dir.name) / channel

    # Creating directories temp_root/functions, temp_root/docs and marketplace_root/functions/(development or master):
    temp_root.mkdir(parents=True)
    temp_docs.mkdir(parents=True)
    marketplace_dir.mkdir(parents=True, exist_ok=True)

    if _verbose:
        print_file_tree("Source project structure", source_dir)
        print_file_tree("Current marketplace structure", marketplace_dir)

    tags = collect_values_from_items(
        source_dir=source_dir,
        tags_set={"categories", "kind"},
    )
    click.echo("Building tags.json...")
    json.dump(tags, open(marketplace_dir / "tags.json", "w"))

    requirements = get_mock_requirements(source_dir)

    if _verbose:
        click.echo(f"[Temporary project] Done requirements ({', '.join(requirements)})")
    sphinx_quickstart(temp_docs, requirements)

    build_temp_project(source_dir, temp_root)
    build_temp_docs(temp_root, temp_docs)
    patch_temp_docs(source_dir, temp_docs)

    if _verbose:
        print_file_tree("Temporary project structure", temp_root)

    render_html_files(temp_docs)

    change_log = ChangeLog()
    copy_resources(marketplace_dir, temp_docs)

    update_or_create_items(
        source_dir,
        marketplace_dir,
        temp_docs,
        change_log,
        force_update=force_update_items,
    )
    build_catalog_json(
        marketplace_dir=marketplace_dir,
        catalog_path=(marketplace_root / "catalog.json"),
        change_log=change_log,
    )
    build_catalog_json(
        marketplace_dir=marketplace_dir,
        catalog_path=(marketplace_dir / "catalog.json"),
        change_log=change_log,
        in_channel_directory=False,
        with_assets=True,
    )

    if _verbose:
        print_file_tree("Resulting marketplace structure", marketplace_dir)

    write_index_html(marketplace_root)
    write_change_log(marketplace_root / "README.md", change_log)


def print_file_tree(title: str, path: Union[str, Path]):
    click.echo(f"\n\n -- {title}:")
    path = Path(path)
    lines = ["---------------------------------", f"\t{path.resolve()}"]
    for file in path.iterdir():
        lines.append("\t|")
        lines.append(f"\t|__ {file.name}")
        if file.is_dir():
            for sub_path in file.iterdir():
                lines.append("\t|\t|")
                lines.append(f"\t|\t|__ {sub_path.name}")
    lines.append("---------------------------------")
    click.echo("\n".join(lines))
    click.echo("\n\n")


def write_change_log(readme_path: Path, change_log: ChangeLog):
    readme_path.touch(exist_ok=True)
    content = open(readme_path, "r").read()
    if change_log.changes_available:
        with open(readme_path, "w") as f:
            compiled_change_log = change_log.compile()
            f.write(compiled_change_log)
            f.write(content)


def write_index_html(marketplace_root: Union[str, Path]):
    marketplace_root = Path(marketplace_root)
    index_path = marketplace_root / "index.html"
    template_path = PROJECT_ROOT / "cli" / "marketplace" / "index.html"

    if index_path.exists():
        index_path.unlink()

    shutil.copy(template_path, index_path)


def copy_resources(marketplace_dir, temp_docs):
    marketplace_static = marketplace_dir / "_static"
    click.echo("Copying static resources...")
    shutil.copytree(
        temp_docs / "_build/_static", marketplace_static, dirs_exist_ok=True
    )


def update_or_create_items(
    source_dir, marketplace_dir, temp_docs, change_log, force_update: bool = False
):
    click.echo("Creating items...")
    for item_dir in PathIterator(root=source_dir, rule=is_item_dir, as_path=True):
        update_or_create_item(
            item_dir, marketplace_dir, temp_docs, change_log, force_update
        )


def build_catalog_json(
    marketplace_dir: Union[str, Path],
    catalog_path: Union[str, Path],
    change_log: ChangeLog,
    in_channel_directory: bool = True,
    with_assets: bool = False,
):
    """
    Building JSON catalog with all the details of the functions in marketplace
    Each function in the catalog is seperated into different versions of the function,
    and in each version field, there is all the details that concerned the version.

    :param marketplace_dir:         the root directory of the marketplace
    :param catalog_path:            the path to the catalog
    :param change_log:              logger of all changes
    :param in_channel_directory:    if True the catalog will be written relatively to channel,
                                    false is relatively to root (the catalog will contain the channels inside)
    :param with_assets:             if True assets will be added to each version with the relative path to them
                                    (API requirement)
    """
    click.echo("Building catalog.json...")

    marketplace_dir = Path(marketplace_dir)
    channel = marketplace_dir.name
    source = marketplace_dir.parent.name

    catalog = json.load(open(catalog_path, "r")) if catalog_path.exists() else {}

    funcs = catalog
    if in_channel_directory:
        if source not in catalog:
            catalog[source] = {}
        if channel not in catalog[source]:
            catalog[source][channel] = {}
        funcs = catalog[source][channel]

    for source_dir in marketplace_dir.iterdir():
        if not source_dir.is_dir() or source_dir.name in ["_static", "_modules"]:
            continue

        latest_dir = source_dir / "latest"
        latest_yaml = update_item_in_catalog(latest_dir, with_assets)

        # removing hidden function from catalog:
        if latest_yaml["hidden"]:
            change_log.hide_item(source_dir.name)
            funcs.pop(source_dir.name, None)
        else:
            funcs[source_dir.name] = {"latest": latest_yaml}

        for version_dir in source_dir.iterdir():
            version = version_dir.name

            if version != "latest":
                version_yaml = update_item_in_catalog(version_dir, with_assets)
                if not latest_yaml["hidden"]:
                    funcs[source_dir.name][version] = version_yaml

    # Remove deleted directories from catalog:
    for function_dir in funcs.keys():
        if not (marketplace_dir / function_dir).exists():
            change_log.deleted_item(function_dir)
            del funcs[function_dir]

    json.dump(catalog, open(catalog_path, "w"))


def update_item_in_catalog(directory: Path, with_assets: bool) -> dict:
    """
    Updates the item yaml in catalog with and add assets if required
    :param directory:   the version directory of the function
    :param with_assets: add function's assets to support MLRun API
    :return: Updated item yaml dictionary
    """
    source_yaml_path = directory / "src" / "item.yaml"

    item_yaml = yaml.full_load(open(source_yaml_path, "r"))
    item_yaml["generationDate"] = str(item_yaml["generationDate"])
    if with_assets:
        add_assets(item_yaml)
    return item_yaml


def add_assets(item_yaml: dict):
    """
    Adding assets to function's item yaml with the relative path for MLRun API

    :param item_yaml: function's item yaml as a dict
    """
    item_yaml["assets"] = {}
    for asset, template in ASSETS.items():
        if isinstance(template, str) and "{}" not in template:
            item_yaml["assets"][asset] = template
        else:
            key, template = template
            asset_location = item_yaml
            for loc in key.split("."):
                asset_location = asset_location.get(loc)
            if asset_location:
                item_yaml["assets"][asset] = template.format(asset_location)


def update_or_create_item(
    item_dir: Path,
    marketplace_dir: Path,
    temp_docs: Path,
    change_log: ChangeLog,
    force_update: bool = False,
):
    # Copy source directories to target directories, if target already has the directory, archive previous version
    item_yaml = yaml.full_load(open(item_dir / "item.yaml", "r"))
    source_version = item_yaml["version"]
    relative_path = "../../../"

    marketplace_item = marketplace_dir / item_dir.stem
    target_latest = marketplace_item / "latest"
    target_version = marketplace_item / source_version

    if target_version.exists() and not force_update:
        latest_item_yaml = yaml.full_load(
            open(target_latest / "src" / "item.yaml", "r")
        )
        if item_yaml["hidden"] == latest_item_yaml.get("hidden"):
            click.echo("Source version already exists in target directory!")
            return

    documentation_html_name = f"{item_dir.stem}.html"
    example_html_name = f"{item_dir.stem}_example.html"

    build_path = temp_docs / "_build"
    source_html = (
        temp_docs / "_build" / "_modules" / item_dir.stem / f"{item_dir.stem}.html"
    )
    update_html_resource_paths(source_html, relative_path=relative_path)

    documentation_html = build_path / documentation_html_name
    update_html_resource_paths(
        documentation_html,
        relative_path=relative_path,
        with_download=False,
        item_name=item_dir.stem,
    )

    example_html = build_path / example_html_name
    update_html_resource_paths(example_html, relative_path=relative_path)

    latest_src = target_latest / "src"
    version_src = target_version / "src"

    # If its the first source is encountered, copy source to target
    if latest_src.exists():
        shutil.rmtree(latest_src)
        if version_src.exists():
            shutil.rmtree(version_src)
        change_log.update_item(item_dir.stem, source_version, target_version.name)
    else:
        change_log.new_item(item_dir.stem, source_version)

    latest_static = target_latest / "static"
    version_static = target_version / "static"

    shutil.copytree(item_dir, latest_src)
    shutil.copytree(item_dir, version_src)

    latest_static.mkdir(parents=True, exist_ok=True)
    version_static.mkdir(parents=True, exist_ok=True)
    if source_html.exists():
        shutil.copy(source_html, latest_static / f"{item_dir.name}.html")
        shutil.copy(source_html, version_static / f"{item_dir.name}.html")

    if documentation_html.exists():
        shutil.copy(documentation_html, latest_static / "documentation.html")
        shutil.copy(documentation_html, version_static / "documentation.html")

    if example_html.exists():
        shutil.copy(example_html, latest_static / "example.html")
        shutil.copy(example_html, version_static / "example.html")

    templates = PROJECT_ROOT / "cli" / "marketplace"

    source_py_name = item_yaml.get("spec", {}).get("filename", "")
    if source_py_name.endswith(".py") and (item_dir / source_py_name).exists():

        with open((item_dir / source_py_name), "r") as f:
            source_code = f.read()

        render_jinja(
            templates / "python.html",
            latest_static / "source.html",
            {"source_code": source_code},
        )
        render_jinja(
            templates / "python.html",
            version_static / "source.html",
            {"source_code": source_code},
        )

    with open((item_dir / "item.yaml"), "r") as f:
        source_code = f.read()

    render_jinja(
        templates / "yaml.html",
        latest_static / "item.html",
        {"source_code": source_code},
    )
    render_jinja(
        templates / "yaml.html",
        version_static / "item.html",
        {"source_code": source_code},
    )

    with open((item_dir / "function.yaml"), "r") as f:
        source_code = f.read()

    render_jinja(
        templates / "yaml.html",
        latest_static / "function.html",
        {"source_code": source_code},
    )
    render_jinja(
        templates / "yaml.html",
        version_static / "function.html",
        {"source_code": source_code},
    )

    pass


def update_html_resource_paths(
    html_path: Path,
    relative_path: str,
    with_download: bool = True,
    item_name: str = None,
):
    if html_path.exists():
        with open(html_path, "r", encoding="utf8") as html:
            parsed = BeautifulSoup(html.read(), features="html.parser")

        # Update back to docs link (from source page)
        back_to_docs_nodes = parsed.find_all(
            lambda node: "viewcode-back" in node.get("class", "")
        )
        pattern = r"^.*?(?={})"
        for node in back_to_docs_nodes:
            node["href"] = re.sub(
                pattern.format(".html"), "documentation", node["href"]
            )

        # Fix links with relative paths:
        nodes = parsed.find_all(
            lambda node: "_static" in node.get("src", "")
            or "_static" in node.get("href", "")
        )
        for node in nodes:
            key = "href" if "_static" in node.get("href", "") else "src"
            node[key] = re.sub(pattern.format("_static"), relative_path, node[key])

        if with_download:
            nodes = parsed.find_all(lambda node: "_sources" in node.get("href", ""))
            for node in nodes:
                # fix path and remove example from name:
                node[
                    "href"
                ] = f'../{node["href"].replace("_sources", "src").replace("_example", "")}'
        else:
            # Removing download option from documentation:
            nodes = parsed.find_all(
                lambda node: node.name == "a" and "headerbtn" in node.get("class", "")
            )
            for node in nodes:
                if node["href"].endswith(".rst"):
                    node.decompose()

        # Fix links in source page:
        if item_name:
            nodes = parsed.find_all(
                lambda node: node.name == "a" and "_modules" in node.get("href", "")
            )
            for node in nodes:
                node["href"] = node["href"].replace(f"_modules/{item_name}/", "")

        with open(html_path, "w", encoding="utf8") as new_html:
            new_html.write(str(parsed))


def render_html_files(temp_docs):
    cmd = f"-b html {temp_docs} {temp_docs / '_build'}"
    click.echo(f"Rendering HTML... [sphinx {cmd}]")
    sphinx_build_cmd(cmd.split(" "))


def patch_temp_docs(source_dir, temp_docs):
    click.echo("Patching temporary docs...")

    for directory in PathIterator(root=source_dir, rule=is_item_dir):
        directory = Path(directory)
        with open(directory / "item.yaml", "r") as f:
            item = yaml.full_load(f)

        example_file = directory / item["example"]
        if example_file:
            example_file = Path(example_file)
            shutil.copy(
                example_file,
                temp_docs / f"{directory.name}_example{example_file.suffix}",
            )


def build_temp_project(source_dir, temp_root):
    click.echo("[Temporary project] Starting to build project...")

    if _verbose:
        click.echo(f"Source dir: {source_dir}")
        click.echo(f"Temp root: {temp_root}")

    item_count = 0
    for directory in PathIterator(root=source_dir, rule=is_item_dir, as_path=True):
        if _verbose:
            item_count += 1
            click.echo(f"[Temporary project] Now processing: {directory / 'item.yaml'}")

        with open(directory / "item.yaml", "r") as f:
            item = yaml.full_load(f)

        filename = item.get("spec")["filename"]
        temp_dir = temp_root / directory.name
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "__init__.py").touch()

        if filename:
            py_file = directory / filename
            temp_file = temp_dir / py_file.name
            shutil.copy(py_file, temp_file)

    if _verbose:
        click.echo(f"[Temporary project] Done project (item count: {item_count})")


def collect_values_from_items(
    source_dir: Union[Path, str], tags_set: Set[str]
) -> Dict[str, List[str]]:
    """
    Collecting all tags values from item.yaml files.
    If the `with_requirements` flag is on than also collecting requirements from ite.yaml and requirements.txt files.

    :param source_dir:          The source directory that contains all the MLRun functions.
    :param tags_set:            Set of tags to collect from item.yaml files.

    :returns:                   A dictionary contains the tags and requirements.
    """

    tags = {t: set() for t in tags_set}

    click.echo(f"[Temporary project] Starting to collect {', '.join(tags_set)}...")

    # Scanning all directories:
    for directory in PathIterator(root=source_dir, rule=is_item_dir, as_path=True):
        # Getting the values from item.yaml
        item_yaml_values = get_item_yaml_values(item_path=directory, keys=tags_set)
        for key, values in item_yaml_values.items():
            tags[key] = tags[key].union(values)

    for tag_name, values in tags.items():
        click.echo(f"[Temporary project] Done {tag_name} ({', '.join(values)})")

    # Change each value from set to list (to be json serializable):
    tags = {key: list(val) for key, val in tags.items()}

    return tags


def sphinx_quickstart(
    temp_root: Union[str, Path], requirements: Optional[List[str]] = None
):
    """
    Generate required files for a Sphinx project. sphinx-quickstart is an
    interactive tool that asks some questions about your project and then
    generates a complete documentation directory and sample Makefile to be used
    with `sphinx-build`.

    This function creates the conf.py configuration file for Sphinx-build based on
    the conf.template file that located in cli/marketplace.

    :param temp_root:       The project's temporary docs root.
    :param requirements:    The list of requirements generated from `get_mock_requirements`
    """
    click.echo("[Sphinx] Running quickstart...")

    # Executing sphinx-quickstart:
    subprocess.run(
        f"sphinx-quickstart --no-sep -p Marketplace -a Iguazio -l en -r '' {temp_root}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )

    # Editing requirements to the necessary format:
    requirements = requirements or ""
    if requirements:
        requirements = '", "'.join(requirements)
        requirements = f'"{requirements}"'

    # Preparing target for conf.py:
    conf_py_target = temp_root / "conf.py"
    conf_py_target.unlink()

    # Rendering the conf.template with the parameters into conf.py target:
    render_jinja(
        template_path=PROJECT_ROOT / "cli" / "marketplace" / "conf.template",
        output_path=conf_py_target,
        data={
            "sphinx_docs_target": temp_root,
            "repository_url": "https://github.com/mlrun/marketplace",
            "mock_imports": requirements,
        },
    )

    click.echo("[Sphinx] Done quickstart")


def build_temp_docs(temp_root, temp_docs):
    """
    Look recursively in <MODULE_PATH> for Python modules and packages and create
    one reST file with automodule directives per package in the <OUTPUT_PATH>. The
    <EXCLUDE_PATTERN>s can be file and/or directory patterns that will be excluded
    from generation. Note: By default this script will not overwrite already
    created files.

    :param temp_root:   The project's temporary functions root.
    :param temp_docs:   The project's temporary docs root.
    """
    click.echo("[Sphinx] Running autodoc...")

    cmd = f"-F -o {temp_docs} {temp_root}"
    click.echo(f"Building temporary sphinx docs... [sphinx-apidoc {cmd}]")

    sphinx_apidoc_cmd(cmd.split(" "))

    click.echo("[Sphinx] Done autodoc")


if __name__ == "__main__":
    # build_marketplace_cli()
    build_marketplace(
        source_dir="../../../functions",
        marketplace_dir="../../../marketplace",
        verbose=True,
        channel="development",
        force_update_items=True,
    )
