# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import click
import yaml

MARKER_START = "<!-- AUTOGEN:START"
MARKER_END = "<!-- AUTOGEN:END -->"
ASSET_COLUMNS = {
    "functions": ("Name", "Description", "Kind", "Categories"),
    "modules": ("Name", "Description", "Kind", "Categories"),
    "steps": ("Name", "Description", "Class Name", "Categories"),
}

@click.command("update-readme")
@click.option("-c", "--channel", default="master", help="Name of build channel")
@click.option(
    "--asset",
    multiple=True,
    required=True,
    help="Asset types to process (e.g: functions). "
         "Pass multiple: --asset functions --asset modules",
)
@click.option("--check", is_flag=True,
              help="Do not write; exit non‑zero if README(s) would change.")
def update_readme(channel: str, asset: Iterable[str],
                      check: bool) -> None:
    """
    Regenerate the README tables for asset types from their item.yaml files.
    """
    asset_list = list(asset)
    changed_any = False
    touched: list[str] = []

    for t in asset_list:
        columns = ASSET_COLUMNS.get(t, ("Name", "Description", "Kind", "Categories"))
        if check:
            # simulate by reading/writing to a temp string, but easiest is: run update and revert if not checking
            # Instead: compute would-change by comparing strings without writing:
            root = Path(".").resolve()
            asset_dir = root / t
            readme = asset_dir / "README.md"
            rows = _rows_for_asset_type(channel, asset_dir, columns)
            table_md = _build_table_md(rows, columns)
            old = readme.read_text() if readme.exists() else f"# {t.title()}\n\n"
            new = _replace_block(old, table_md)
            if new != old:
                changed_any = True
                touched.append(str(readme))
        else:
            if _update_one(channel, t, columns):
                changed_any = True
                touched.append(str((Path(t) / "README.md").as_posix()))

    if check and changed_any:
        click.echo("README tables are out of date for:")
        for p in touched:
            click.echo(f"  - {p}")
        sys.exit(1)

    # Normal run prints what it updated (no failure)
    if not check:
        if changed_any:
            click.echo("Updated README(s):")
            for p in touched:
                click.echo(f"  - {p}")
        else:
            click.echo("No README changes.")


def _rows_for_asset_type(channel: str, asset_dir: Path, columns) -> list:
    """Scan <asset>/src/*/item.yaml and return table rows."""
    src = asset_dir / "src"
    if not src.exists():
        return []

    rows = []
    for item_yaml in sorted(src.glob("*/item.yaml")):
        asset_name = item_yaml.parent.name
        try:
            data = yaml.safe_load(item_yaml.read_text()) or {}
        except Exception as e:
            raise click.ClickException(f"Failed reading {item_yaml}: {e}") from e

        desc = (data.get("description") or "").strip()
        kind = (data.get("spec", {}).get("kind", "")).strip()
        class_name = (data.get("className", "")).strip()
        cats = data.get("categories") or []
        cats_str = ", ".join(c.strip() for c in cats) if isinstance(cats, list) else str(cats).strip()
        # Link the name to its source directory
        # Construct the relative path from the repo root for the asset
        rel_path = asset_dir.relative_to(Path(".").resolve())
        link = f"[{asset_name}](https://github.com/mlrun/functions/tree/{channel}/{rel_path}/src/{asset_name})"
        row = []
        for col in columns:
            if col == "Name":
                row.append(link)
            elif col == "Description":
                row.append(desc)
            elif col == "Kind":
                row.append(kind)
            elif col == "Class Name":
                row.append(class_name)
            elif col == "Categories":
                row.append(cats_str)
            else:
                row.append("")
        rows.append(tuple(row))

    rows.sort(key=lambda r: r[0].lower())
    return rows


def _build_table_md(rows, columns) -> str:
    if not rows:
        return "_No items found_"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join((cell or "").replace("\n", " ").strip() for cell in r) + " |")
    return "\n".join(lines)


def _replace_block(readme_text: str, new_block: str) -> str:
    si = readme_text.find(MARKER_START)
    ei = readme_text.find(MARKER_END)
    if si == -1 or ei == -1 or ei < si:
        # Append a section if markers are missing
        section = (
            f"\n## Catalog\n\n"
            f"{MARKER_START} (do not edit below) -->\n"
            f"{new_block}\n"
            f"{MARKER_END}\n"
        )
        return readme_text.rstrip() + "\n\n" + section

    # Ensure we keep the whole START marker line up to "-->"
    start_close = readme_text.find("-->", si)
    if start_close == -1:
        start_close = si + len(MARKER_START)
        readme_text = readme_text[:start_close] + " -->" + readme_text[start_close:]
        start_close = readme_text.find("-->", si)
    start_close += 3  # include the "-->"

    return readme_text[:start_close] + "\n" + new_block + "\n" + readme_text[ei:]


def _update_one(channel: str, asset_type: str, columns) -> bool:
    """Generate/replace the table in <asset_type>/README.md. Return True if changed."""
    root = Path(".").resolve()
    asset_dir = root / asset_type
    readme = asset_dir / "README.md"

    rows = _rows_for_asset_type(channel, asset_dir, columns)
    table_md = _build_table_md(rows, columns)
    old = readme.read_text() if readme.exists() else f"# {asset_type.title()}\n\n"
    new = _replace_block(old, table_md)

    if new != old:
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(new)
        return True
    return False
