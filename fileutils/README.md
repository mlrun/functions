# File Utilities Function

## open_archive

Open a remote zip archive into a local target folder, [ource](file_utils.py).

Usage example:

```python
# load function from Github
xfn = mlrun.import_function('https://raw.githubusercontent.com/yjb-ds/functions/master/fileutils/open_archive.yaml')

# configute it: mount on iguazio fabric, set as interactive (return stdout)
xfn.apply(mlrun.mount_v3io())
xfn.interactive = True

# create and run the task
images_path = '/User/mlrun/functions/images'
open_archive_task = mlrun.NewTask('download', handler='open_archive', 
                                  params={'target_dir': images_path},
                                  inputs={'archive_url': 'http://iguazio-sample-data.s3.amazonaws.com/catsndogs.zip'})

# run
run = xfn.run(open_archive_task)
```

Output:

```
[mlrun] 2019-10-28 22:30:31,825 starting run download uid=2ec277feb3b644e2a45c92ce8cb2537a  -> http://mlrun-db:8080
[mlrun] 2019-10-28 22:30:31,858 using in-cluster config.
[mlrun] 2019-10-28 22:30:31,882 Pod download-qpnsd created
.....
[mlrun] 2019-10-28 22:30:41,911 starting run download uid=2ec277feb3b644e2a45c92ce8cb2537a  -> http://mlrun-db:8080
[mlrun] 2019-10-28 22:30:41,994 downloading http://iguazio-sample-data.s3.amazonaws.com/catsndogs.zip to local tmp
[mlrun] 2019-10-28 22:30:43,750 Verified directories
[mlrun] 2019-10-28 22:30:43,750 Extracting zip
[mlrun] 2019-10-28 22:31:03,671 extracted archive to /User/mlrun/examples/images
type result.show() to see detailed results/progress or use CLI:
!mlrun get run --uid 2ec277feb3b644e2a45c92ce8cb2537a 
[mlrun] 2019-10-28 22:31:03,699 run executed, status=completed

```
## arc_to_parquet

Retrieve a remote archive and save locally as a parquet file, [source](file_utils.py)

Usage example:

```python
# load function from Github
xfn = mlrun.import_function('https://raw.githubusercontent.com/yjb-ds/functions/master/fileutils/arc_to_parquet.yaml')

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