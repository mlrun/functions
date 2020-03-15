## arc_to_parquet

Retrieve a remote archive and save locally as a parquet file, [source](arc_to_parquet.py)

Usage example:

```python
import mlrun, os
mlrun.mlconf.mlrundb = 'http://mlrun-api:8080'

FUNCTIONS_LIB = 'https://raw.githubusercontent.com/yjb-ds/functions/refac'

# load arc_to_parquet function from Github
func = mlrun.code_to_function(os.path.join(FUNCTIONS_LIB, 'arc_to_parquet/function.yaml'))

# configure function: mount on Iguazio data fabric, set as interactive (return stdout)
func.apply(mlrun.mount_v3io())

# create and run the task
images_path = '/User/functions/arc_to_parquet/images'
archive = 'XXXXXXXXXXXXXXXXXXXXXXXX'

arc_to_parq_task = mlrun.NewTask('arc2parq', 
                                 handler='arc_to_parquet',
                                 params={
                                     'archive_url': archive,
                                     'key'        : 'raw_data_as_parquet'},
                                artifact_path=images_path)
# run
run = func.run(arc_to_parq_task)
```

Output:

```
```