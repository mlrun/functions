from pathlib import Path
import shutil

from mlrun import code_to_function, import_function

ARTIFACTS_PATH = 'artifacts'
CONTENT_PATH = 'content/data/images'
ARCHIVE_URL = "https://s3.wasabisys.com/iguazio/data/cats-vs-dogs/cats-vs-dogs-labeling-demo.zip"



def _delete_outputs(paths):
    for path in paths:
        if Path(path).is_dir():
            shutil.rmtree(path)


# def test_open_archive():
#     fn = code_to_function(name='test_open_archive',
#                           filename="open_archive.py",
#                           handler="open_archive",
#                           kind="local",
#                           )
#     fn.spec.command = "open_archive.py"
#     fn.run(inputs={'archive_url': ARCHIVE_URL}
#            )
#     assert Path(CONTENT_PATH).is_dir()
#     _delete_outputs({'artifacts', 'runs', 'schedules', 'content'})


def test_open_archive_import_function():
    fn = import_function("function.yaml")
    fn.run(inputs={'archive_url': ARCHIVE_URL}
           ,local=True)
    assert Path(CONTENT_PATH).is_dir()
    _delete_outputs({'artifacts', 'runs', 'schedules', 'content'})


