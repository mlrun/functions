# **Training Functions**

## `sklearn-classifer with Dask`

Run any scikit-learn compatible classifier or list of classifiers with Dask

### steps

1. **Generate a scikit-learn model configuration** using the `model_pkg_class` parameter
   * input a package and class name, for example, `sklearn.linear_model.LogisticRegression`  
   * mlrun will find the class and instantiate a copy using default parameters  
   * You can modify both the model class and the fit methods
2. **Get a sample of data** from a data source
   * select a random sample of rows using a negative integer
   * select consecutive rows using a positive integer
3. **Split the data** into train, validation, and test sets 
   * the test set is saved as an artifact and never seen again until testing
4. **Train the model** 
5. **pickle / serialize the model**
   * models can be pickled or saved as json
6. **Evaluate the model**
   * a custom evaluator can be provided, see function doc for details

'''markdown
Train a sklearn classifier with Dask
    
    :param context:                 Function context.
    :param dataset:                 Raw data file.
    :param model_pkg_class:         Model to train, e.g, "sklearn.ensemble.RandomForestClassifier", 
                                    or json model config.
    :param label_column:            (label) Ground-truth y labels.
    :param train_validation_size:   (0.75) Train validation set proportion out of the full dataset.
    :param sample:                  (1.0) Select sample from dataset (n-rows/% of total), randomzie rows as default.
    :param models_dest:             (models) Models subfolder on artifact path.
    :param test_set_key:            (test_set) Mlrun db key of held out data in artifact store.
    :param plots_dest:              (plots) Plot subfolder on artifact path.
    :param dask_key:                (dask key) Key of dataframe in dask client "datasets" attribute.
    :param dask_persist:            (False) Should the data be persisted (through the `client.persist`)
    :param scheduler_key:           (scheduler) Dask scheduler configuration, json also logged as an artifact.
    :param file_ext:                (parquet) format for test_set_key hold out data
    :param random_state:            (42) sklearn seed
'''

### TODO

1. Add cross validation methods
2. Improve dask efficiency by calling dask data frame (not from pandas)
3. Log dataset artifact as dask data frame 
4. Add values imputer (instead of drop na)