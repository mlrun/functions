# churn server

the `churn-server` function was created as part of the **[churn demo](https://github.com/yjb-ds/demo-churn)**.  A model server was needed that could combine the static model which answers the binary classification question "is this client churned or not-churned?" and the more dynamic model, which tries to add a time dimension to the prediction by providing an esdtimate of when and with what certainty churn events are likely to occur.

the function `coxph_trainer` will output multiple models within a nested directory structire starting at `models_dest`:
* the coxph model is stored at `models_dest/cox`
* the [kaplan-meier](https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator) model at `models_dest/cox/km`

each one of these pickled models stores all of the meta-data, vector and table estimates, including projections and scenarios

with only slight modification, a more generic version of this server would enable its application in the domains of **[predictive maintenance](https://docs.microsoft.com/en-us/archive/msdn-magazine/2019/may/machine-learning-using-survival-analysis-for-predictive-maintenance)**, **[health](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3227332/)**, **finance** and **insurance** to name a few.

**note**

a small file `encode-data.csv` can be find in the root of this function folder, it is used to test the server.