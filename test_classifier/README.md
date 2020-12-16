# **Testing Functions**

## `sklearn-classifer`

Test one or more classifier models against held-out dataset
Using held-out test features, evaluates the performance of the estimated model
Can be part of a kubeflow pipeline as a test step that is run post EDA and
training/validation cycles.

```markdown

:param context:            the function context
:param models_path:        artifact models representing a file or a folder
:param test_set:           test features and labels
:param label_column:       column name for ground truth labels
:param score_method:       for multiclass classification
:param plots_dest:         dir for test plots
:param model_evaluator:    NOT IMPLEMENTED: specific method to generate eval, passed in as string
                               or available in this folder
:param predictions_column: column name for the predictions column on the resulted artifact
:param model_update:       (True) update model, when running as stand alone no need in update
```