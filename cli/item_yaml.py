import click
from cli.path_iterator import PathIterator
from cli.helpers import is_item_dir
import yaml


@click.command()
@click.option("-r", "--root-directory", default=".", help="Path to root directory")
@click.option("-v", "--version", help="update version number in function item yaml")
@click.option("-mv", "--mlrun-version", help="update mlrun version in function item.yaml")
@click.option("-p", "--platform-version", help="update platform version in function item.yaml")
def update_functions_yaml(root_directory: str,
                          version: str,
                          mlrun_version: str,
                          platform_version: str):
    if not root_directory:
        click.echo("-r/--root-directory is required")
        exit(1)

    item_iterator = PathIterator(root=root_directory, rule=is_item_dir, as_path=True)
    for inner_dir in item_iterator:
        item_yaml = "item.yaml"
        if (inner_dir / item_yaml).exists():
            path = str(inner_dir)+"/"+item_yaml
            stream = open(path, 'r')
            data = yaml.load(stream)
            if version:
                data['version'] = version
            if mlrun_version:
                data['mlrunVersion'] = mlrun_version
            if platform_version:
                data['platformVersion'] = platform_version
            print(data)
            with open(path, 'w') as yaml_file:
                yaml_file.write(yaml.dump(data, default_flow_style=False))
