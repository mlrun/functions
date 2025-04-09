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
import shutil
import os

from mlrun import code_to_function, import_function

ARTIFACTS_PATH = 'artifacts'
CONTENT_PATH = 'content/data/images'
ARCHIVE_URL = "https://s3.wasabisys.com/iguazio/data/cats-vs-dogs/cats-vs-dogs-labeling-demo.zip"



def _delete_outputs(paths):
    for path in paths:
        if Path(path).is_dir():
            shutil.rmtree(path)


def test_open_archive():
    fn = code_to_function(name='test_open_archive',
                          filename="open_archive.py",
                          handler="open_archive",
                          kind="local",
                          )
    fn.spec.command = "open_archive.py"
    fn.run(inputs={'archive_url': ARCHIVE_URL},
                 params={'key': 'test_archive', 'target_path': os.getcwd() + '/content/'},
                 local=True)
    
    assert Path(CONTENT_PATH).is_dir()
    _delete_outputs({'artifacts', 'runs', 'schedules', 'content'})


def test_open_archive_import_function():
    fn = import_function("function.yaml")
    run = fn.run(inputs={'archive_url': ARCHIVE_URL},
                 params={'key': 'test_archive', 'target_path': os.getcwd() + '/content/'},
                 local=True)
    print(run.to_dict())
    assert (run.artifact('test_archive'))
    _delete_outputs({'artifacts', 'runs', 'schedules', 'content'})