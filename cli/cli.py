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

from cli.functions.function_to_item import function_to_item_cli
from cli.functions.item_to_function import item_to_function_cli
from cli.marketplace.build import build_marketplace_cli
from cli.functions.new_function_item import new_item
from cli.common.test_suite import test_suite
from cli.common.item_yaml import update_functions_yaml


@click.group()
def cli():
    pass


cli.add_command(new_item)
cli.add_command(item_to_function_cli, name="item-to-function")
cli.add_command(function_to_item_cli, name="function-to-item")
cli.add_command(test_suite, name="run-tests")
cli.add_command(build_marketplace_cli, name="build-marketplace")
cli.add_command(update_functions_yaml, name="update-functions-yaml")

if __name__ == "__main__":
    cli()
