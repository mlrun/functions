import json
from pathlib import Path
from typing import Union

import click as click
import yaml
from mlrun import import_function

from cli.helpers import PROJECT_ROOT
from cli.path_iterator import PathIterator


@click.command()
@click.option(
    "-r", "--root-dir", default=PROJECT_ROOT, help="Path to root project directory"
)
def create_legacy_catalog(root_dir: Union[str, Path]):
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise RuntimeError("Root directory must be a directory")

    catalog = {}

    file_list = Path(root_dir).glob("**/*.yaml")

    for file in sorted(file_list, key=lambda f: str(f)):
        file = file.resolve()
        click.echo(f"Now inspecting file: {file}")

        if file.is_file():
            try:
                fn = import_function(str(file))
            except Exception as e:
                click.echo(f"failed to load func {file}, {e}")
                continue

            if not fn.kind or fn.kind in ["", "local", "handler"]:
                click.echo(f"illegal function or kind in {file}, kind={fn.kind}")
                continue

            if fn.metadata.name in catalog:
                entry = catalog[fn.metadata.name]
            else:
                file_dir = file.parent
                notebook_iterator = PathIterator(
                    root=file_dir,
                    rule=lambda p: p.name.endswith(".ipynb"),
                    as_path=True,
                )
                notebooks = list(notebook_iterator)
                doc_file = file_dir if not notebooks else file_dir / notebooks[0]
                entry = {
                    "description": fn.spec.description,
                    "categories": fn.metadata.categories,
                    "kind": fn.kind,
                    "docfile": str(doc_file.resolve()),
                    "versions": {},
                }

            entry["versions"][fn.metadata.tag or "latest"] = str(file)
            print(fn.metadata.name, entry)
            catalog[fn.metadata.name] = entry

    with open("catalog.yaml", "w") as fp:
        yaml.dump(catalog, fp)

    with open("catalog.json", "w") as fp:
        json.dump(catalog, fp)

    mdheader = """# Functions hub 

This functions hub is intended to be a centralized location for open source contributions of function components.  
These are functions expected to be run as independent mlrun pipeline compnents, and as public contributions, 
it is expected that contributors follow certain guidelines/protocols (please chip-in).

## Functions
"""

    with open(root_dir / "README.md", 'w') as fp:
        fp.write(mdheader)
        rows = []
        for k, v in catalog.items():
            kind = v['kind']
            if kind == 'remote':
                kind = 'nuclio'
            row = [f"[{k}]({v['docfile']})", kind, v['description'], ', '.join(v['categories'] or [])]
            rows.append(row)

        text = gen_md_table(['function', 'kind', 'description', 'categories'], rows)
        fp.write(text)


def gen_md_table(header, rows=None):
    rows = [] if rows is None else rows

    def gen_list(items=None):
        items = [] if items is None else items
        out = '|'
        for i in items:
            out += ' {} |'.format(i)
        return out

    out = gen_list(header) + '\n' + gen_list(len(header) * ['---']) + '\n'
    for r in rows:
        out += gen_list(r) + '\n'
    return out




