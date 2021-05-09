from mlrun import code_to_function
import os

ARCHIVE = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
ARTIFACTS_PATH = 'artifacts'


def test_arc_to_parquet():
    cwd = os.getcwd()
    fn = code_to_function(name='test_arc_to_parquet',
                          filename="arc_to_parquet.py",
                          handler="arc_to_parquet",
                          kind="job",
                          )
    fn.run(params={'archive_url': ARCHIVE,
                   'key': 'HIGGS'},
           artifact_path=cwd,
           local=True

           )
