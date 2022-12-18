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
from pathlib import Path
from typing import Optional, Callable, Union


class PathIterator:
    """Creates a file paths iterator on a given root path

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
