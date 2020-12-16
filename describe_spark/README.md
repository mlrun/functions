# WIP - Spark Describe Function with MLRun (non-sparkoperator)

## Run .py file Using Spark
### Steps:
1. Deploy spark-operator on the cluster (create service from dashboard).
   This is required at this stage in order to create a configmap for the daemon.
2. In Jupyter:
   Save the followin code under my.py in fuse (in this case /v3io/users/admin/my.py):

```python
#!/usr/local/bin/python
 
# Locate v3iod:
from subprocess import run
run(["/bin/bash", "/etc/config/v3io/v3io-spark-operator.sh"])
 
# The pyspark code:
import os
from pyspark.sql import SparkSession
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages mysql:mysql-connector-java:5.1.39 pyspark-shell"
 
spark = (SparkSession.builder.appName("Spark JDBC to Databases - ipynb")
                            .config("spark.driver.extraClassPath", "/v3io/users/admin/mysql-connector-java-5.1.45.jar")
                            .config("spark.executor.extraClassPath", "/v3io/users/admin/mysql-connector-java-5.1.45.jar")
                            .getOrCreate())

dfMySQL = (spark.read.format("jdbc")     
                     .option("url", "jdbc:mysql://mysql-rfam-public.ebi.ac.uk:4497/Rfam")
                     .option("dbtable", "Rfam.family")     
                     .option("user", "rfamro")    
                     .option("password", "")    
                     .option("driver", "com.mysql.jdbc.Driver")     
                     .load())
 
dfMySQL.write.format("io.iguaz.v3io.spark.sql.kv").mode("overwrite").option("key", "rfam_id").save("v3io://users/admin/frommysql")
 
spark.stop()
```

3. Make sure that your script has execution permissions.
4. Execute the following block in a notebook:

```python
from mlrun import new_function
from mlrun.platforms.iguazio import mount_v3io, mount_v3iod
import os
image_name = 'iguazio/shell:' + os.environ.get("IGZ_VERSION")
run = new_function(name='my-spark', image=image_name , command='/v3io/users/admin/my.py', kind='job', mode='pass')
run.apply(mount_v3io(name="v3io-fuse", remote="/", mount_path="/v3io"))
run.apply(mount_v3iod(namespace="default-tenant", v3io_config_configmap="spark-operator-v3io-config"))
run.run(artifact_path="/User/artifacts")
```
---

## Create Simple Read CSV Function Using Spark
Please refer to the read_csv_spark notebook

---

## Create Describe Function Using Spark
Generates profile reports from an Apache Spark DataFrame. 
Based on pandas_profiling, but for Spark's DataFrames instead of pandas.

For each column the following statistics - if relevant for the column type - are presented:

* `Essentials:` type, unique values, missing values
* `Quantile statistics:` minimum value, Q1, median, Q3, maximum, range, interquartile range
* `Descriptive statistics:` mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
* `Most frequent values:` for categorical data

```
Function params

:param context:               Function context.
:param dataset:               Raw data file (currently needs to be a local file located in v3io://User/bigdata)
:param bins:                  Number of bin in histograms
:param describe_extended:     (True) set to False if the aim is to get a simple .describe() infomration
```

* All operations are done efficiently, which means that **no** Python UDFs or .map() transformations are used at all; 
* only Spark SQL's Catalyst is used for the retrieval of all statistics.

---
### TODO:
1. Add plots
2. Add ability to generte html report
