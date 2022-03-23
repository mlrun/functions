import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Union, Optional, Set, Dict, Tuple, List

import click
import yaml
from bs4 import BeautifulSoup
from sphinx.cmd.build import main as sphinx_build_cmd
from sphinx.ext.apidoc import main as sphinx_apidoc_cmd

from cli.helpers import (
    is_item_dir,
    render_jinja,
    PROJECT_ROOT,
    get_requirements_from_txt,
    get_item_yaml_values,
    remove_version_constraints,
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
        marketplace_dir,
        source_name,
        temp_dir,
        channel,
        verbose,
    )


def build_marketplace(
    source_dir: str,
    marketplace_dir: str,
    source_name: Optional[str] = None,
    temp_dir: str = "/tmp",
    channel: str = "development",
    verbose: bool = False,
):
    """Main entry point to marketplace building

    :param source_dir: Path to the source directory to build the marketplace from
    :param marketplace_dir: Path to marketplace directory
    :param source_name: Name of source, if not provided, name of source directory will be used instead
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

    # Creating directories temp_root/functions, temp_root/docs and marketplace_root/functions/(development or master):
    temp_root.mkdir(parents=True)
    temp_docs.mkdir(parents=True)
    marketplace_dir.mkdir(parents=True, exist_ok=True)

    if _verbose:
        print_file_tree("Source project structure", source_dir)
        print_file_tree("Current marketplace structure", marketplace_dir)

    requirements = build_tags_json_and_collect_requirements(
        source_dir=source_dir,
        marketplace_dir=marketplace_root,
        tags_set={"categories", "kind"},
    )

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
    build_catalog_json(
        marketplace_dir=marketplace_dir,
        catalog_path=(marketplace_root / "catalog.json"),
    )
    build_catalog_json(
        marketplace_dir=marketplace_dir,
        catalog_path=(marketplace_dir / "catalog.json"),
        with_functions_legacy=False,
        artifacts=True,
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


def copy_static_resources(marketplace_dir, temp_docs):
    marketplace_static = marketplace_dir / "_static"
    if not marketplace_static.exists():
        click.echo("Copying static resources...")
        shutil.copytree(temp_docs / "_build/_static", marketplace_static)


def update_or_create_items(source_dir, marketplace_dir, temp_docs, change_log):
    click.echo("Creating items...")
    for item_dir in PathIterator(root=source_dir, rule=is_item_dir, as_path=True):
        update_or_create_item(item_dir, marketplace_dir, temp_docs, change_log)


def create_artifacts_section(version_dir: Path, function_name: str):
    artifacts = {}

    assert (version_dir / "src").exists() and (
        version_dir / "static"
    ).exists(), "Each version must contain src and static directories"

    files_to_add = {
        "doc": version_dir / "static" / "documentation.html",
        "object": version_dir / "src" / "function.yaml",
        "example": version_dir / "src" / f"{function_name}.ipynb",
        "source": version_dir / "src" / f"{function_name}.py",
    }

    for field, path in files_to_add.items():
        if path.exists():
            artifacts[field] = "/".join(path.parts[-2:])

    return artifacts


def build_catalog_json(
    marketplace_dir: Union[str, Path],
    catalog_path: Union[str, Path],
    with_functions_legacy: bool = True,
    artifacts: bool = False,
):
    click.echo("Building catalog.json...")

    marketplace_dir = Path(marketplace_dir)
    channel = marketplace_dir.name
    source = marketplace_dir.parent.name

    catalog = json.load(open(catalog_path, "r")) if catalog_path.exists() else {}

    if with_functions_legacy:
        if source not in catalog:
            catalog[source] = {}
        if channel not in catalog[source]:
            catalog[source][channel] = {}
    for source_dir in marketplace_dir.iterdir():
        if not source_dir.is_dir() or source_dir.name == "_static":
            continue

        latest_dir = source_dir / "latest"
        source_yaml_path = latest_dir / "src" / "item.yaml"

        latest_yaml = yaml.full_load(open(source_yaml_path, "r"))
        latest_yaml["generationDate"] = str(latest_yaml["generationDate"])

        latest_version = latest_yaml["version"]
        if artifacts:
            latest_yaml["artifacts"] = create_artifacts_section(
                version_dir=latest_dir, function_name=source_dir.name
            )
        if with_functions_legacy:
            catalog[source][channel][source_dir.name] = {"latest": latest_yaml}
        else:
            catalog[source_dir.name] = {"latest": latest_yaml}
        for version_dir in source_dir.iterdir():
            version = version_dir.name

            if version != "latest" and version != latest_version:
                version_yaml_path = version_dir / "src" / "item.yaml"
                version_yaml = yaml.full_load(open(version_yaml_path, "r"))
                version_yaml["generationDate"] = str(version_yaml["generationDate"])
                if artifacts:
                    version_yaml["artifacts"] = create_artifacts_section(
                        version_dir=version_dir, function_name=source_dir.name
                    )
                if with_functions_legacy:
                    catalog[source][channel][source_dir.name][version] = version_yaml
                else:
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

    documentation_html_name = f"{item_dir.stem}.html"
    example_html_name = f"{item_dir.stem}_example.html"

    build_path = temp_docs / "_build"

    documentation_html = build_path / documentation_html_name
    update_html_resource_paths(documentation_html, relative_path="../../../")

    example_html = build_path / example_html_name
    update_html_resource_paths(example_html, relative_path="../../../")

    latest_src = target_latest / "src"
    version_src = target_version / "src"

    # If its the first source is encountered, copy source to target
    if latest_src.exists():
        shutil.rmtree(latest_src)
        change_log.update_item(item_dir.stem, source_version, target_version.name)
    else:
        change_log.new_item(item_dir.stem, source_version)

    latest_static = target_latest / "static"
    version_static = target_version / "static"

    shutil.copytree(item_dir, latest_src)
    shutil.copytree(item_dir, version_src)

    latest_static.mkdir(parents=True, exist_ok=True)
    version_static.mkdir(parents=True, exist_ok=True)

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


def update_html_resource_paths(html_path: Path, relative_path: str):
    if html_path.exists():
        with open(html_path, "r", encoding="utf8") as html:
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

        nodes = parsed.find_all(lambda node: "_sources" in node.get("href", ""))
        for node in nodes:
            node[
                "href"
            ] = f'../{node["href"].replace("_sources", "src").replace("_example", "")}'

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


def build_tags_json_and_collect_requirements(
    source_dir: Union[Path, str],
    marketplace_dir: Union[Path, str],
    tags_set: Set[str],
) -> Set[str]:
    """
    Building tags.json by collecting values from item.yaml files
    and collecting requirements from txt and item.yaml files.

    :param source_dir:          The source directory that contains all the MLRun functions.
    :param marketplace_dir:     The marketplace directory, tags.json will be placed in it.
    :param tags_set:            Set of tags to collect from item.yaml files.

    :returns:                   A set of all the requirements.
    """
    tags, requirements = collect_values(source_dir=source_dir, tags_set=tags_set)

    click.echo("Building tags.json...")
    json.dump(tags, open(marketplace_dir / "tags.json", "w"))
    return requirements


def collect_values(
    source_dir: Union[Path, str], tags_set: Set[str], with_requirements: bool = True
) -> Union[Dict[str, List[str]], Tuple[Dict[str, List[str]], Set[str]]]:
    """
    Collecting all tags values from item.yaml files.
    If the `with_requirements` flag is on than also collecting requirements from ite.yaml and requirements.txt files.

    :param source_dir:          The source directory that contains all the MLRun functions.
    :param tags_set:            Set of tags to collect from item.yaml files.
    :param with_requirements:   flag for collecting requirements from item.yaml and requirements.txt files.

    :returns:                   A dictionary contains the tags and requirements.
    """

    tags = {t: set() for t in tags_set}
    requirements = None

    if with_requirements:
        requirements = {"mlrun"}
        tags_set.update({"requirements"})

    values_to_collect = ", ".join(tags_set)
    click.echo(f"[Temporary project] Starting to collect {values_to_collect}...")

    # Scanning all directories:
    for directory in PathIterator(root=source_dir, rule=is_item_dir, as_path=True):
        # Getting the values from item.yaml
        item_yaml_values = get_item_yaml_values(item_path=directory, keys=tags_set)
        for key, values in item_yaml_values.items():
            if key == "requirements":
                # Getting the requirements from the requirements.txt file:
                txt_requirements = get_requirements_from_txt(
                    requirements_path=directory
                )
                parsed_requirements = remove_version_constraints(
                    values.union(txt_requirements)
                )
                requirements.update(parsed_requirements)
            else:
                tags[key] = tags[key].union(values)

    if _verbose:
        if with_requirements:
            click.echo(
                f"[Temporary project] Done requirements ({', '.join(requirements)})"
            )
        for tag_name, values in tags.items():
            click.echo(f"[Temporary project] Done {tag_name} ({', '.join(values)})")

    # Change each value from set to list (to be json serializable):
    tags = {key: list(val) for key, val in tags.items()}

    if with_requirements:
        return tags, requirements

    return tags


def sphinx_quickstart(
    temp_root: Union[str, Path], requirements: Optional[Set[str]] = None
):
    """
    Generate required files for a Sphinx project. sphinx-quickstart is an
    interactive tool that asks some questions about your project and then
    generates a complete documentation directory and sample Makefile to be used
    with `sphinx-build`.

    This function creates the conf.py configuration file for Sphinx-build based on
    the conf.template file that located in cli/marketplace.

    :param temp_root:       The project's temporary docs root.
    :param requirements:    The set of requirements generated from `collect_temp_requirements`
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
    build_marketplace("../../", "../../../marketp")
