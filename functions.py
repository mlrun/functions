import click

from common.item_to_function import item_to_function
from common.marketplace.build_docs import build_docs
from common.new_item import new_item
from common.test_suite import test_suite


@click.group()
def cli():
    pass


cli.add_command(new_item)
cli.add_command(item_to_function)
cli.add_command(test_suite)
cli.add_command(build_docs)

if __name__ == "__main__":
    cli()
