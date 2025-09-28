import sys
from pathlib import Path
from datetime import datetime
import click
from jinja2 import Environment, FileSystemLoader

TEMPLATES = {
    "function": "cli/utils/function_item_template.yaml.j2",
    "module": "cli/utils/module_item_template.yaml.j2",
}


@click.command()
@click.argument("type", type=click.Choice(TEMPLATES.keys()))
@click.argument("name")
def generate_item_yaml(type, name):
    """
    Generate an item.yaml file from a template.

    TYPE: one of {function, module}
    NAME: name of the function/module (also directory name)
    """
    # Construct the target path
    path = Path(f"{type}s/src/{name}").resolve()
    output_file = path / "item.yaml"

    if output_file.exists():
        click.echo(f"Error: {output_file} already exists.", err=True)
        sys.exit(1)

    if not path.exists():
        click.echo(f"Error: {path} does not exist.", err=True)
        sys.exit(1)

    # Render parameters
    params = {
        "example": f"{name}.ipynb",
        "generationDate": datetime.utcnow().strftime("%Y-%m-%d"),
        "name": name,
        "filename": f"{name}.py",
    }

    # Load and render template
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(TEMPLATES[type])
    rendered = template.render(params)

    output_file.write_text(rendered)
    click.echo(f"Created {output_file}")


if __name__ == "__main__":
    generate_item_yaml()