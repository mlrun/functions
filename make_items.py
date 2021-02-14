from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from mlrun import import_function


@dataclass
class Spec:
    filename: str = ""
    handler: str = ""
    requirements: List[str] = field(default_factory=list)
    kind: str = ""
    image: str = ""


@dataclass
class Maintainer:
    name: str = ""
    email: str = ""


@dataclass
class Item:
    api_version: str = "v1"
    org: str = "Iguazio"
    name: str = ""
    version: str = ""
    mlrun_version: str = ""
    platform_version: str = ""
    description: str = ""
    doc: str = ""
    example: str = ""
    icon: str = ""
    url: str = ""
    generationDate: str = ""
    categories: List[str] = field(default_factory=list)
    labels: Dict[str, Union[str, int, float]] = field(default_factory=dict)
    spec: Spec = field(default_factory=Spec)
    maintainers: List[Maintainer] = field(default_factory=list)
    marketplaceType: str = ""

    @classmethod
    def from_yaml(cls, yaml_path: Path):
        model = import_function(str(yaml_path.absolute()))
        item = Item(
            name=model.metadata.name or "",
            version=model.metadata.tag or "0.0.1",
            mlrun_version="0.5.4",
            platform_version="2.10.0",
            description=model.spec.description or "",
            doc="",
            example=_locate_ipynb_file(yaml_path.parent) or "",
            icon="",
            url="",
            generationDate=str(datetime.utcnow()),
            categories=model.metadata.categories or [],
            labels=model.metadata.labels or {},
            spec=Spec(
                filename=_locate_py_file(yaml_path.parent) or "",
                handler=model.spec.default_handler or "",
                requirements=[],
                kind=model.kind or "",
                image=_get_image(model),
            ),
            maintainers=[],
        )
        return item


def _snake_case_to_lower_camel_case(string: str) -> str:
    if "_" not in string:
        return string
    else:
        components = string.split("_")
        return components[0] + "".join(c.title() for c in components[1:])


def _locate_py_file(dir_path: Path) -> Optional[str]:
    default_py_file = dir_path / "function.py"

    if default_py_file.exists:
        return "function.py"

    py_file = list(filter(lambda d: d.suffix == ".py", dir_path.iterdir()))

    if len(py_file) > 1:
        raise RuntimeError(
            "Failed to infer business logic python file name, found multiple python files"
        )
    elif len(py_file) == 1:
        return py_file[0].name

    return None


def _locate_ipynb_file(dir_path: Path) -> Optional[str]:
    notebook_file = list(filter(lambda d: d.suffix == ".ipynb", dir_path.iterdir()))

    if len(notebook_file) > 1:
        return None

    return notebook_file[0].name


def _get_image(model):
    try:
        return model.spec.image
    except Exception:
        try:
            return model.spec.build.base_image
        except Exception:
            return ""


def _function_yaml_to_item_yaml(function_yaml: Path) -> dict:
    # Construct item
    item = Item.from_yaml(function_yaml)
    # Make change casing to fit common standards
    item = {_snake_case_to_lower_camel_case(k): v for k, v in asdict(item).items()}
    return item


def _make_item(path: Path):
    item = _function_yaml_to_item_yaml(path / "function.yaml")
    with open(path / "item.yaml", "w") as f:
        yaml.dump(item, f)


if __name__ == "__main__":
    for path in map(lambda p: p.parent.absolute(), Path().glob("*/function.yaml")):
        print(f"Creating item.yaml for {path}")
        _make_item(path)
