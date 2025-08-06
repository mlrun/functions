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
import click
from functions.cli.utils.path_iterator import PathIterator
from functions.cli.utils.helpers import is_item_dir
import yaml
import datetime


@click.command()
@click.option("-r", "--root-directory", default=".", help="Path to root directory")
@click.option("-v", "--version", help="update version number in function item yaml")
@click.option("-mv", "--mlrun-version", help="update mlrun version in function item.yaml")
@click.option("-p", "--platform-version", help="update platform version in function item.yaml")
@click.option("-d", "--date-time", help="update date-time in function item.yaml")
def update_functions_yaml(root_directory: str,
                          version: str,
                          mlrun_version: str,
                          platform_version: str,
                          date_time: str):
    if not root_directory:
        click.echo("-r/--root-directory is required")
        exit(1)

    item_iterator = PathIterator(root=root_directory, rule=is_item_dir, as_path=True)
    for inner_dir in item_iterator:
        item_yaml = "item.yaml"
        if (inner_dir / item_yaml).exists():
            path = str(inner_dir)+"/"+item_yaml
            stream = open(path, 'r')
            data = yaml.load(stream=stream, Loader=yaml.FullLoader)
            if version:
                data['version'] = version
            if mlrun_version:
                data['mlrunVersion'] = mlrun_version
            if platform_version:
                data['platformVersion'] = platform_version
            if date_time:
                data['generationDate'] = datetime.datetime.now().strftime('%Y-%m-%d:%H-%M')
            print(data)
            with open(path, 'w') as yaml_file:
                yaml_file.write(yaml.dump(data, default_flow_style=False))
