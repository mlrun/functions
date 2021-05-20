from mlrun.datastore import DataItem
from mlrun.artifacts import get_model
from cloudpickle import load
from mlrun.mlutils.models import eval_class_model


def cox_test(
        context,
        models_path: DataItem,
        test_set: DataItem,
        label_column: str,
        plots_dest: str = "plots",
        model_evaluator=None
) -> None:
    """Test one or more classifier models against held-out dataset

    Using held-out test features, evaluates the peformance of the estimated model

    Can be part of a kubeflow pipeline as a test step that is run post EDA and
    training/validation cycles

    :param context:         the function context
    :param model_file:      model artifact to be tested
    :param test_set:        test features and labels
    :param label_column:    column name for ground truth labels
    :param score_method:    for multiclass classification
    :param plots_dest:      dir for test plots
    :param model_evaluator: WIP: specific method to generate eval, passed in as string
                            or available in this folder
    """
    xtest = test_set.as_df()
    ytest = xtest.pop(label_column)

    model_file, model_obj, _ = get_model(models_path.url, suffix='.pkl')
    model_obj = load(open(str(model_file), "rb"))

    try:
        # there could be different eval_models, type of model (xgboost, tfv1, tfv2...)
        if not model_evaluator:
            # binary and multiclass
            eval_metrics = eval_class_model(context, xtest, ytest, model_obj)

        # just do this inside log_model?
        model_plots = eval_metrics.pop("plots")
        model_tables = eval_metrics.pop("tables")
        for plot in model_plots:
            context.log_artifact(plot, local_path=f"{plots_dest}/{plot.key}.html")
        for tbl in model_tables:
            context.log_artifact(tbl, local_path=f"{plots_dest}/{plot.key}.csv")

        context.log_results(eval_metrics)
    except:
        # dummy log:
        context.log_dataset("cox-test-summary", df=model_obj.summary, index=True, format="csv")
        context.logger.info("cox tester not implemented")
