from mlrun import code_to_function , import_function
from pathlib import Path
import os
import shutil

ARTIFACTS_PATH = 'artifacts'
DATA_URL = "https://s3.wasabisys.com/iguazio/data/market-palce/arc_to_parquet/higgs-sample.csv.gz"


def arc_to_parquet():
    from mlrun import import_function
    from mlrun.platforms import auto_mount

    archive_func = import_function('hub://arc_to_parquet')
    archive_run = archive_func.run(handler="arc_to_parquet",
                        params={"key":"rent", "stats": True, "file_ext":"csv"},
                        inputs={"archive_url": DATA_URL},
                        artifact_path=os.getcwd() + '/artifacts'
                        , local=True
                        )


def sklearn_classifier(run):
    cwd = os.getcwd()
    file_path = str(Path(cwd).parent.absolute()) + "/sklearn_classifier/sklearn_classifier.py"
    fn = code_to_function(name='test_sklearn_classifier',
                          filename=file_path,
                          handler="train_model",
                          kind="local",
                          )
    fn.spec.command = file_path
    fn.run(params={
        "sample"                 : -5_000, # 5k random rows,
        "model_pkg_class"        : "sklearn.ensemble.RandomForestClassifier",
        "label_column"           : "interest_level",
        "CLASS_n_estimators"     : 100,
        "CLASS_min_samples_leaf" : 1,
        "CLASS_n_jobs"           : -1,
        "CLASS_oob_score"        : True},
           handler="train_model",
           inputs={"dataset": run.outputs["rent"]},
           artifact_path='artifacts'
           # , local=True
           )


def test_run_local_feature_selection():
    arc_run = arc_to_parquet()
    #sk_run =  sklearn_classifier(arc_run)
    # labels = "interest_level"
    #
    # fn = code_to_function(name='test_run_local_feature_perms',
    #                       filename="feature_perms.py",
    #                       handler="permutation_importance",
    #                       kind="local",
    #                       )
    # fi_perms = fn.run(params={"labels": labels,
    #                     "plots_dest": "plots"},
    #     inputs={"model": model, "dataset": data},
    #     artifact_path=artifact_path)


def test_tust():
    from mlrun import import_function
    from mlrun.platforms import auto_mount

    train = import_function('hub://sklearn_classifier')
    #.apply(auto_mount())

    train_run = train.run(
                          inputs={"dataset": DATA_URL},
                          params={
                              "sample": -5_000,  # 5k random rows,
                              "model_pkg_class": "sklearn.ensemble.RandomForestClassifier",
                              "label_column": "interest_level",
                              "CLASS_n_estimators": 100,
                              "CLASS_min_samples_leaf": 1,
                              "CLASS_n_jobs": -1,
                              "CLASS_oob_score": True},
                          local=True)