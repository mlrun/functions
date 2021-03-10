## pandas_profiling_report

Creates an html report with various graphs/statistics/correlations for a given dataset. See sample report [here](https://pandas-profiling.github.io/pandas-profiling/examples/master/titanic/titanic_report.html). Link to GitHub page [here](https://github.com/pandas-profiling/pandas-profiling).


Usage example:

```python
import mlrun, os
mlrun.mlconf.dbpath = 'http://mlrun-api:8080'

# Load pandas_profiling_report function from Github
func = mlrun.import_function("hub://pandas_profiling_report").apply(mlrun.mount_v3io())

# Build MLRun image (only needs to be run once)
func.deploy()

# Create task
data = 'https://iguazio-sample-data.s3.amazonaws.com/datasets/iris_dataset.csv'

task = NewTask(name="pandas-profiling-report", 
               inputs={"data": DATA_URL})

# Run task on cluster
run = func.run(task, artifact_path='/User/artifacts')
```
