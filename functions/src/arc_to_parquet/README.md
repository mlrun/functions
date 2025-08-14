## arc_to_parquet

Retrieve a remote archive and save locally as a parquet file, [source](arc_to_parquet.py)

Usage example:

```python
import mlrun, os
mlrun.mlconf.dbpath = 'http://mlrun-api:8080'
mlrun.mlconf.hub_url = '/User/functions/{name}/function.yaml'

# load arc_to_parquet function from Github
func = mlrun.import_function("hub://arc_to_parquet").apply(mlrun.mount_v3io())

# create and run the task
archive = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

arc_to_parq_task = mlrun.NewTask(name='tasks - acquire remote', 
                                 params={'archive_url': archive,
                                         'key'        : 'HIGGS'})
# run
run = func.run(arc_to_parq_task, artifact_path='/User/artifacts')
```

Output:

```
```