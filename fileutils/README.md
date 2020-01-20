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