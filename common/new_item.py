from datetime import datetime
from pathlib import Path

import click


@click.command()
@click.option(
    "-p", "--path", help="Path to directory in which a new item.yaml will be created"
)
@click.option("-o", "--override", is_flag=True, help="Override if already exists")
def new_item(path: str, override: bool):
    path = Path(path) / "item.yaml"

    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    elif path.exists() and not override:
        click.echo(
            f"{path / 'item.yaml'} already exists, set [-o, --override] to override"
        )
        exit(1)

    with open(path, "w") as f:
        f.write(
            f"""
apiVersion: v1
categories: []         # List of category names
description: ''        # Short description
doc: ''                # Path to README.md if exists
example: ''            # Path to examole notebook
generationDate: {str(datetime.utcnow())}
icon: ''               # Path to icon file
labels: {{}}             # Key values label pairs
maintainers: []        # List of maintainers
mlrunVersion: ''       # Function’s MLRun version requirement, should follow python’s versioning schema
name: ''               # Function name
platformVersion: ''    # Function’s Iguazio version requirement, should follow python’s versioning schema
spec:
  filename: ''         # Implementation file
  handler: ''          # Handler function name
  image: ''            # Base image name
  kind: ''             # Function kind
  requirements: []     # List of Pythonic library requirements
url: ''                # ???
version: 0.0.1         # Function version, should follow standard semantic versioning schema
"""
        )


if __name__ == "__main__":
    new_item()
