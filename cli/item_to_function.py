from pathlib import Path
from typing import Optional
import os
import click
from mlrun import code_to_function
from yaml import full_load
from cli.path_iterator import PathIterator
from cli.helpers import (
    is_item_dir
)


@click.command()
@click.option(
    "-i", "--item-path", help="Path to item.yaml file or a directory containing one"
)
@click.option("-o", "--output-path", help="Path to code_to_function output")
@click.option("-r", "--root-directory", help="Path to root directory containing multiple functions")
def item_to_function(item_path: str, output_path: Optional[str] = None, root_directory: Optional[str] = None):
    if root_directory is None:
        item_to_function_from_path(item_path, output_path)
    else:
        item_iterator = PathIterator(root=root_directory, rule=is_item_dir, as_path=True)

        for inner_dir in item_iterator:
            # Iterate individual files in each directory
            for inner_file in inner_dir.iterdir():
                if inner_file.is_file() and inner_file.name == 'item.yaml':
                    inner_file_dir = os.path.dirname(str(inner_file))
                    output_file_path = inner_file_dir + "/" + "gen_function.yaml"
                    print(inner_file_dir)
                    try:
                        item_to_function_from_path(str(inner_file_dir), output_file_path)
                    except Exception :
                        print("failed to generate yaml for {}".format(str(inner_dir)))


def item_to_function_from_path(item_path: str, output_path: Optional[str] = None):
    item_path = Path(item_path)
    base_path = ""
    if item_path.is_dir():
        if (item_path / "item.yaml").exists():
            base_path = str(item_path)
            item_path = item_path / "item.yaml"

        else:
            raise FileNotFoundError(f"{item_path} does not contain a item.yaml file")
    elif not item_path.exists():
        raise FileNotFoundError(f"{item_path} not found")

    item_yaml = full_load(open(item_path, "r"))

    filename = item_yaml.get("spec", {}).get("filename")

    code_output = ""
    if filename.endswith(".ipynb"):
        code_output = Path(filename)
        code_output = code_output.parent / f"{code_output.stem}.py"
    code_file_name = get_filename(base_path, item_yaml)

    function_object = code_to_function(
        name=item_yaml["name"],
        filename=code_file_name,
        handler=item_yaml.get("spec", {}).get("handler"),
        kind=item_yaml.get("spec", {}).get("kind"),
        code_output=code_output,
        image=item_yaml.get("spec", {}).get("image"),
        description=item_yaml.get("description", ""),
        requirements=item_yaml.get("spec", {}).get("requirements"),
        categories=item_yaml.get("categories", []),
        labels=item_yaml.get("labels", {}),
    )

    if output_path is None:
        return function_object

    output_path = Path(output_path)

    if output_path.is_dir():
        output_path = output_path / "function.yaml"

    if not output_path.parent.exists():
        output_path.mkdir()

    function_object.export(target=str(output_path.resolve()))


def get_filename(base_path, item_yaml):
    filename = item_yaml.get("spec", {}).get("filename")
    if filename is '':
        filename = base_path + "/" + item_yaml.get("example")
    else:
        filename = base_path + "/" + item_yaml.get("spec", {}).get("filename")
    return filename


if __name__ == "__main__":
    item_to_function()
