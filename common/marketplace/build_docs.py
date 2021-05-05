import json
import shutil
import uuid
from pathlib import Path
from typing import Union

import click
import yaml
from bs4 import BeautifulSoup
from sphinx.cmd.build import main as sphinx_build_cmd
from sphinx.ext.apidoc import main as sphinx_apidoc_cmd

from common.helpers import is_item_dir, render_jinja_file, PROJECT_ROOT
from common.marketplace.changelog import ChangeLog
from common.path_iterator import PathIterator


@click.command()
@click.option("-s", "--source-dir", help="Path to the source directory")
@click.option("-t", "--target-dir", help="Path to output directory")
@click.option(
    "-T", "--temp-dir", default="/tmp", help="Path to intermediate build directory"
)
@click.option("-c", "--channel", default="master", help="Name of build channel")
def build_docs(source_dir: str, target_dir: str, temp_dir: str, channel: str):
    root_base = Path(temp_dir) / uuid.uuid4().hex
    temp_root = root_base / "functions"
    temp_docs = root_base / "docs"

    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()

    target_channel = target_dir / channel

    temp_root.mkdir(parents=True)
    temp_docs.mkdir(parents=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_channel.mkdir(parents=True, exist_ok=True)

    click.echo(f"Temporary working directory: {root_base}")

    build_temp_project(source_dir, temp_root)
    build_temp_docs(temp_docs, temp_root)
    patch_temp_docs(source_dir, temp_docs, temp_root)
    render_html_files(temp_docs)

    change_log = ChangeLog()
    copy_static_resources(target_dir, temp_docs)

    update_or_create_items(change_log, source_dir, target_channel, temp_docs)
    build_catalog_json(target_channel)

    write_change_log(target_dir / "README.md", change_log)


def write_change_log(readme: Path, change_log: ChangeLog):
    readme.touch(exist_ok=True)
    content = open(readme, "r").read()
    with open(readme, "w") as f:
        if change_log.changes_available:
            compiled_change_log = change_log.compile()
            f.write(compiled_change_log)
        f.write(content)


def copy_static_resources(target_dir, temp_docs):
    target_static = target_dir / "_static"
    if not target_static.exists():
        click.echo("Copying static resources...")
        shutil.copytree(temp_docs / "_build/_static", target_static)


def update_or_create_items(change_log, source_dir, target_dir, temp_docs):
    click.echo("Creating items...")
    for directory in PathIterator(root=source_dir, rule=is_item_dir, as_path=True):
        update_or_create_item(directory, target_dir, temp_docs, change_log)


def build_catalog_json(target_dir: Union[str, Path]):
    click.echo("Building catalog.json...")
    target_dir = Path(target_dir)
    catalog_path = target_dir / "catalog.json"
    catalog = json.load(open(catalog_path, "r")) if catalog_path.exists() else {}

    for source_dir in target_dir.iterdir():
        if not source_dir.is_dir() or source_dir.name == "_static":
            continue

        latest_dir = source_dir / "latest"
        source_yaml_path = latest_dir / "item.yaml"

        latest_yaml = yaml.full_load(open(source_yaml_path, "r"))
        latest_yaml["generationDate"] = str(latest_yaml["generationDate"])

        latest_version = latest_yaml["version"]

        catalog[source_dir.name] = {"latest": latest_yaml}
        for version_dir in source_dir.iterdir():
            version = version_dir.name

            if version != "latest" and version != latest_version:
                version_yaml_path = version_dir / "item.yaml"
                version_yaml = yaml.full_load(open(version_yaml_path, "r"))
                version_yaml["generationDate"] = str(version_yaml["generationDate"])
                catalog[source_dir.name][version] = version_yaml

    json.dump(catalog, open(catalog_path, "w"))


def update_or_create_item(
    source_dir: Path, target: Path, temp_docs: Path, change_log: ChangeLog
):
    # Copy source directories to target directories, if target already has the directory, archive previous version
    source_yaml = yaml.full_load(open(source_dir / "item.yaml", "r"))
    source_version = source_yaml["version"]

    target_dir = target / source_dir.stem
    target_latest = target_dir / "latest"
    target_version = target_dir / source_version

    if target_version.exists():
        click.echo("Source version already exists in target directory!")
        return

    build_path = temp_docs / "_build"
    source_html_name = f"{source_dir.stem}.html"
    example_html_name = f"{source_dir.stem}_example.html"

    source_html = build_path / source_html_name
    update_html_resource_paths(source_html, relative_path="../../../")

    example_html = build_path / example_html_name
    update_html_resource_paths(example_html, relative_path="../../../")

    # If its the first source is encountered, copy source to target
    if target_dir.exists():
        shutil.rmtree(target_latest)
        change_log.update_item(source_dir.stem, source_version, target_version.name)
    else:
        change_log.new_item(source_dir.stem, source_version)

    shutil.copytree(source_dir, target_latest)
    shutil.copytree(source_dir, target_version)

    if source_html.exists():
        shutil.copy(source_html, target_latest / source_html_name)
        shutil.copy(source_html, target_version / source_html_name)

    if example_html.exists():
        shutil.copy(example_html, target_latest / example_html_name)
        shutil.copy(example_html, target_version / example_html_name)


def update_html_resource_paths(html_path: Path, relative_path: str):
    if html_path.exists():
        with open(html_path, "r") as html:
            parsed = BeautifulSoup(html.read(), features="html.parser")

        nodes = parsed.find_all(
            lambda node: node.name == "link" and "_static" in node.get("href", "")
        )
        for node in nodes:
            node["href"] = f"{relative_path}{node['href']}"

        nodes = parsed.find_all(
            lambda node: node.name == "script"
            and node.get("src", "").startswith("_static")
        )
        for node in nodes:
            node["src"] = f"{relative_path}{node['src']}"

        with open(html_path, "w") as new_html:
            new_html.write(str(parsed))


def render_html_files(temp_docs):
    cmd = f"-b html {temp_docs} {temp_docs / '_build'}"
    click.echo(f"Rendering HTML... [sphinx {cmd}]")
    sphinx_build_cmd(cmd.split(" "))


def patch_temp_docs(source_dir, temp_docs, temp_root):
    click.echo("Patching temporary docs...")
    for directory in PathIterator(root=source_dir, rule=is_item_dir):
        directory = Path(directory)
        with open(directory / "item.yaml", "r") as f:
            item = yaml.full_load(f)

        example_file = directory / item["example"]
        shutil.copy(example_file, temp_docs / f"{directory.name}_example.ipynb")

    conf_py_target = temp_docs / "conf.py"
    conf_py_target.unlink()

    render_jinja_file(
        template_path=PROJECT_ROOT / "common" / "marketplace" / "conf.template",
        output_path=conf_py_target,
        data={
            "sphinx_docs_target": temp_root,
            "repository_url": "https://github.com/mlrun/marketplace",
        },
    )


def build_temp_project(source_dir, temp_root):
    click.echo("Building temporary project...")
    for directory in PathIterator(root=source_dir, rule=is_item_dir, as_path=True):
        with open(directory / "item.yaml", "r") as f:
            item = yaml.full_load(f)

        py_file = directory / item.get("spec")["filename"]

        temp_dir = temp_root / directory.name
        temp_dir.mkdir(parents=True, exist_ok=True)

        (temp_dir / "__init__.py").touch()
        shutil.copy(py_file, temp_dir / py_file.name)


def build_temp_docs(temp_docs, temp_root):
    cmd = f"-F -o {temp_docs} {temp_root}"
    click.echo(f"Building temporary sphinx docs... [sphinx-apidoc {cmd}]")
    sphinx_apidoc_cmd(cmd.split(" "))


if __name__ == "__main__":
    build_docs()
