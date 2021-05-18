import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Union, Optional, Set

import click
import yaml
from bs4 import BeautifulSoup
from sphinx.cmd.build import main as sphinx_build_cmd
from sphinx.ext.apidoc import main as sphinx_apidoc_cmd

from cli.helpers import (
    is_item_dir,
    render_jinja_file,
    PROJECT_ROOT,
    get_item_yaml_requirements,
)
from cli.marketplace.changelog import ChangeLog
from cli.path_iterator import PathIterator

_verbose = False


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
def build_marketplace_cli(
    source_dir: str,
    source_name: str,
    marketplace_dir: str,
    temp_dir: str,
    channel: str,
    verbose: bool,
):
    build_marketplace(
        source_dir,
        source_name,
        marketplace_dir,
        temp_dir,
        channel,
        verbose,
    )


def build_marketplace(
    source_dir: str,
    source_name: str,
    marketplace_dir: str,
    temp_dir: str,
    channel: str,
    verbose: bool,
):
    """Main entry point to marketplace building

    :param source_dir: Path to the source directory to build the marketplace from
    :param source_name: Name of source, if not provided, name of source directory will be used instead
    :param marketplace_dir: Path to marketplace directory
    :param temp_dir: Path to intermediate directory, used to build marketplace resources,
    if not provided '/tmp/<random_uuid>' will be used
    :param channel: The name of the marketplace channel to write to
    :param verbose: When True, additional debug information will be written to stdout
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

    temp_root.mkdir(parents=True)
    temp_docs.mkdir(parents=True)
    marketplace_dir.mkdir(parents=True, exist_ok=True)

    if _verbose:
        print_file_tree("Source project structure", source_dir)
        print_file_tree("Current marketplace structure", marketplace_dir)

    requirements = collect_temp_requirements(source_dir)
    sphinx_quickstart(temp_docs, requirements)

    build_temp_project(source_dir, temp_root)
    build_temp_docs(temp_root, temp_docs)
    patch_temp_docs(source_dir, temp_docs)

    if _verbose:
        print_file_tree("Temporary project structure", temp_root)

    render_html_files(temp_docs)

    change_log = ChangeLog()
    copy_static_resources(marketplace_dir, temp_docs)

    update_or_create_items(source_dir, marketplace_dir, temp_docs, change_log)
    build_catalog_json(marketplace_dir)

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
    with open(readme_path, "w") as f:
        if change_log.changes_available:
            compiled_change_log = change_log.compile()
            f.write(compiled_change_log)
        f.write(content)


def write_index_html(marketplace_root: Union[str, Path]):
    marketplace_root = Path(marketplace_root)
    items = []
    item_num = 0
    for source_dir in marketplace_root.iterdir():
        if source_dir.is_file():
            continue

        source_name = source_dir.name
        for channel_dir in source_dir.iterdir():
            if channel_dir.is_file():
                continue

            channel_name = channel_dir.name

            catalog_path = channel_dir / "catalog.json"
            catalog = json.load(open(catalog_path))

            for item_name, item in catalog.items():
                item_num += 1
                remote_base_url = (
                    Path(source_name) / channel_name / item_name / "latest"
                )
                latest = item["latest"]
                version = latest["version"]
                generation_date = latest["generationDate"]
                file_name = latest["spec"]["filename"].split(".")[0]
                file_url = remote_base_url / f"{file_name}.html"
                example_url = remote_base_url / f"{file_name}_example.html"
                items.append(
                    {
                        "item_num": item_num,
                        "source_name": source_name,
                        "channel_name": channel_name,
                        "item_name": item_name,
                        "version": version,
                        "generation_date": generation_date,
                        "file_url": file_url,
                        "example_url": example_url,
                    }
                )

    if items:
        index_path = marketplace_root / "index.html"
        if index_path.exists():
            index_path.unlink()
        render_jinja_file(
            template_path=PROJECT_ROOT / "cli" / "marketplace" / "index.template",
            output_path=index_path,
            data={"items": items},
        )


def copy_static_resources(marketplace_dir, temp_docs):
    marketplace_static = marketplace_dir / "_static"
    if not marketplace_static.exists():
        click.echo("Copying static resources...")
        shutil.copytree(temp_docs / "_build/_static", marketplace_static)


def update_or_create_items(source_dir, marketplace_dir, temp_docs, change_log):
    click.echo("Creating items...")
    for item_dir in PathIterator(root=source_dir, rule=is_item_dir, as_path=True):
        update_or_create_item(item_dir, marketplace_dir, temp_docs, change_log)


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
    item_dir: Path, marketplace_dir: Path, temp_docs: Path, change_log: ChangeLog
):
    # Copy source directories to target directories, if target already has the directory, archive previous version
    item_yaml = yaml.full_load(open(item_dir / "item.yaml", "r"))
    source_version = item_yaml["version"]

    marketplace_item = marketplace_dir / item_dir.stem
    target_latest = marketplace_item / "latest"
    target_version = marketplace_item / source_version

    if target_version.exists():
        click.echo("Source version already exists in target directory!")
        return

    build_path = temp_docs / "_build"
    source_html_name = f"{item_dir.stem}.html"
    example_html_name = f"{item_dir.stem}_example.html"

    source_html = build_path / source_html_name
    update_html_resource_paths(source_html, relative_path="../../")

    example_html = build_path / example_html_name
    update_html_resource_paths(example_html, relative_path="../../")

    # If its the first source is encountered, copy source to target
    if marketplace_item.exists():
        shutil.rmtree(target_latest)
        change_log.update_item(item_dir.stem, source_version, target_version.name)
    else:
        change_log.new_item(item_dir.stem, source_version)

    shutil.copytree(item_dir, target_latest)
    shutil.copytree(item_dir, target_version)

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


def patch_temp_docs(source_dir, temp_docs):
    click.echo("Patching temporary docs...")

    for directory in PathIterator(root=source_dir, rule=is_item_dir):
        directory = Path(directory)
        with open(directory / "item.yaml", "r") as f:
            item = yaml.full_load(f)

        example_file = directory / item["example"]
        shutil.copy(example_file, temp_docs / f"{directory.name}_example.ipynb")


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

        py_file = directory / item.get("spec")["filename"]

        temp_dir = temp_root / directory.name
        temp_dir.mkdir(parents=True, exist_ok=True)

        (temp_dir / "__init__.py").touch()
        shutil.copy(py_file, temp_dir / py_file.name)

    if _verbose:
        click.echo(f"[Temporary project] Done project (item count: {item_count})")


def collect_temp_requirements(source_dir) -> Set[str]:
    click.echo("[Temporary project] Starting to collect requirements...")
    requirements = set()

    for directory in PathIterator(root=source_dir, rule=is_item_dir, as_path=True):
        item_requirements = get_item_yaml_requirements(directory)
        for item_requirement in item_requirements:
            requirements.add(item_requirement)

    if _verbose:
        click.echo(f"[Temporary project] Done requirements ({', '.join(requirements)})")

    return requirements


def sphinx_quickstart(
    temp_root: Union[str, Path], requirements: Optional[Set[str]] = None
):
    click.echo("[Sphinx] Running quickstart...")

    subprocess.run(
        f"sphinx-quickstart --no-sep -p Marketplace -a Iguazio -l en -r '' {temp_root}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )

    requirements = requirements or ""
    if requirements:
        requirements = '", "'.join(requirements)
        requirements = f'"{requirements}"'

    conf_py_target = temp_root / "conf.py"
    conf_py_target.unlink()

    render_jinja_file(
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
    click.echo("[Sphinx] Running autodoc...")

    cmd = f"-F -o {temp_docs} {temp_root}"
    click.echo(f"Building temporary sphinx docs... [sphinx-apidoc {cmd}]")

    sphinx_apidoc_cmd(cmd.split(" "))

    click.echo("[Sphinx] Done autodoc")


if __name__ == "__main__":
    build_marketplace_cli()
