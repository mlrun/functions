## arc_to_parquet

Retrieve a remote archive and save locally as a parquet file, [source](arc_to_parquet.py)

Usage example:

```python
# load function from Github
func = mlrun.new_function(
        command='https://raw.githubusercontent.com/yjb-ds/functions/master/arc_to_parquet/function.py',
        image=f'yjb-ds/mlrun-base:v0.4.5',
        kind='job')

func.export('function.yaml')

# configure function: mount on Iguazio data fabric, set as interactive (return stdout)
xfn.apply(mlrun.mount_v3io())
xfn.interactive = True

# create and run the task
images_path = '/User/mlrun/functions/images'
archive = 'https://fpsignals-public.s3.amazonaws.com/x_test_50.csv.gz'

arc_to_parq_task = mlrun.NewTask('arc2parq', 
                                 handler='arc_to_parquet',
                                 params={
                                     'target_path': target_path,
                                     'name'       : 'x_test_50.csv',
                                     'key'        : 'raw_data',
                                     'archive_url': archive})
# run
run = xfn.run(open_archive_task)
```

Output:

```
[mlrun] 2020-01-09 21:28:47,515 starting run arc2parq uid=ed20cbdcddb3473882507594f69e6180  -> http://mlrun-api:8080
[mlrun] 2020-01-09 21:29:03,735 destination file does not exist, downloading
[mlrun] 2020-01-09 21:29:03,873 saved table to /User/mlrun/functions/parquet/x_test_50.parquet
[mlrun] 2020-01-09 21:29:03,873 logging /User/mlrun/functions/parquet/x_test_50.parquet to context

[mlrun] 2020-01-09 21:29:03,898 run executed, status=completed
...
to track results use .show() or .logs() or in CLI: 
!mlrun get run ed20cbdcddb3473882507594f69e6180  , !mlrun logs ed20cbdcddb3473882507594f69e6180 
[mlrun] 2020-01-09 21:29:06,867 run executed, status=completed
```