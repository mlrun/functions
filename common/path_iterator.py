from pathlib import Path
from typing import Optional, Callable, Union


class PathIterator:
    """ Creates a file paths iterator on a given root path
    
    :param root: Root directory
    :param rule: A function returning a boolean that is used to indicate filtering
    :param recursive: If true, will iterate all files recursively, False by default
    :param absolute: If true, will return the absolute file path, True by default
    """

    def __init__(
        self,
        root: Union[str, Path],
        rule: Optional[Callable[[Path], bool]] = None,
        recursive: bool = False,
        absolute: bool = True,
        as_path: bool = False,
    ) -> None:
        self.root = Path(root)
        self.absolute = absolute
        self.rule = rule or (lambda _: True)
        self.recursive = recursive
        self.as_path = as_path

    def __iter__(self):
        iterator = self.root.rglob("*") if self.recursive else self.root.iterdir()
        for path in iterator:
            if self.rule(path):
                path = path.resolve() if self.absolute else path
                if self.as_path:
                    yield path
                else:
                    yield str(path)
