import shutil
from optparse import OptionParser
from pathlib import Path


def new_item(path: str, exists_ok: bool = False):
    path = Path(path) / "item.yaml"

    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    elif path.exists() and not exists_ok:
        print(f"{path / 'item.yaml'} already exists, set [-e, --exists-ok] to override")
        exit(1)

    shutil.copy("./common/item_template.yaml", path)


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "-p",
        "--path",
        help="Path to directory in which a new item.yaml will be created",
    )

    parser.add_option(
        "-e",
        "--exists-ok",
        help="Override if already exists [False]/True",
        default=False,
    )

    options, args = parser.parse_args()
    new_item(path=options.path, exists_ok=bool(options.exists_ok))
