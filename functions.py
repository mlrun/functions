import click

from cli.create_legacy_catalog import create_legacy_catalog
from cli.item_to_function import item_to_function
from cli.marketplace.build_docs import build_docs
from cli.new_item import new_item
from cli.test_suite import test_suite


@click.group()
def cli():
    pass


cli.add_command(new_item)
cli.add_command(item_to_function)
cli.add_command(test_suite)
cli.add_command(build_docs)
cli.add_command(create_legacy_catalog)

if __name__ == "__main__":
    cli()
