from pathlib import Path
import shutil
from mlrun import code_to_function, import_function
#Testing the CI with dummy commit 14
AGGREGATE_PATH = "artifacts/aggregate.pq"
DATA = "https://s3.wasabisys.com/iguazio/data/market-palce/aggregate/metrics.pq"


def test_run_local_aggregate():
    fn = code_to_function(name='test_hello_world',
                          filename="hello_world.py",
                          handler="hello_world",
                          kind="local",
                          )
    fn.spec.command = 'hello_world.py'
    fn.run(
           )


def test_import_function_aggregate():
    fn = import_function("function.yaml")
    fn.run(local=True)

