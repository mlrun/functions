# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import os

import matplotlib.pyplot as plt
import mlrun
import mlrun.common.schemas
import mlrun.feature_store as fs
import numpy as np
import pandas as pd
import seaborn as sns
from mlrun.artifacts import PlotArtifact
from mlrun.datastore.targets import ParquetTarget
# MLRun utils
from mlrun.utils.helpers import create_class
# Feature selection strategies
from sklearn.feature_selection import SelectFromModel, SelectKBest
# Scale feature scoresgit st
from sklearn.preprocessing import MinMaxScaler
# SKLearn estimators list
from sklearn.utils import all_estimators

DEFAULT_STAT_FILTERS = ["f_classif", "mutual_info_classif", "chi2", "f_regression"]
DEFAULT_MODEL_FILTERS = {
    "LinearSVC": "LinearSVC",
    "LogisticRegression": "LogisticRegression",
    "ExtraTreesClassifier": "ExtraTreesClassifier",
}


def _clear_current_figure():
    """
    Clear matplotlib current figure.
    """
    plt.cla()
    plt.clf()
    plt.close()


def show_values_on_bars(axs, h_v="v", space=0.4):
    def _show_on_single_plot(ax_):
        if h_v == "v":
            for p in ax_.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax_.text(_x, _y, value, ha="center")
        elif h_v == "h":
            for p in ax_.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax_.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def plot_stat(context, stat_name, stat_df):
    _clear_current_figure()

    # Add chart
    ax = plt.axes()
    stat_chart = sns.barplot(
        x=stat_name,
        y="index",
        data=stat_df.sort_values(stat_name, ascending=False).reset_index(),
        ax=ax,
    )
    plt.tight_layout()

    for p in stat_chart.patches:
        width = p.get_width()
        plt.text(
            5 + p.get_width(),
            p.get_y() + 0.55 * p.get_height(),
            "{:1.2f}".format(width),
            ha="center",
            va="center",
        )

    context.log_artifact(
        PlotArtifact(f"{stat_name}", body=plt.gcf()),
        local_path=os.path.join("plots", "feature_selection", f"{stat_name}.html"),
    )
    _clear_current_figure()


def feature_selection(
    context,
    df_artifact,
    k: int = 5,
    min_votes: float = 0.5,
    label_column: str = None,
    stat_filters: list = None,
    model_filters: dict = None,
    max_scaled_scores: bool = True,
    sample_ratio: float = None,
    output_vector_name: float = None,
    ignore_type_errors: bool = False,
    is_feature_vector: bool = False,
):
    """
    Applies selected feature selection statistical functions or models on our 'df_artifact'.

    Each statistical function or model will vote for it's best K selected features.
    If a feature has >= 'min_votes' votes, it will be selected.

    :param context:             the function context.
    :param df_artifact:         dataframe to pass as input.
    :param k:                   number of top features to select from each statistical
                                function or model.
    :param min_votes:           minimal number of votes (from a model or by statistical
                                function) needed for a feature to be selected.
                                Can be specified by percentage of votes or absolute
                                number of votes.
    :param label_column:        ground-truth (y) labels.
    :param stat_filters:        statistical functions to apply to the features
                                (from sklearn.feature_selection).
    :param model_filters:       models to use for feature evaluation, can be specified by
                                model name (ex. LinearSVC), formalized json (contains 'CLASS',
                                'FIT', 'META') or a path to such json file.
    :param max_scaled_scores:   produce feature scores table scaled with max_scaler.
    :param sample_ratio:        percentage of the dataset the user whishes to compute the feature selection process on.
    :param output_vector_name:  creates a new feature vector containing only the identifies features.
    :param ignore_type_errors:  skips datatypes that are neither float nor int within the feature vector.
    :param is_feature_vector:   bool stating if the data is passed as a feature vector.
    """
    stat_filters = stat_filters or DEFAULT_STAT_FILTERS
    model_filters = model_filters or DEFAULT_MODEL_FILTERS
    # Check if df.meta is valid, if it is, look for a feature vector
    if df_artifact.meta:
        if df_artifact.meta.kind == mlrun.common.schemas.ObjectKind.feature_vector:
            is_feature_vector = True

    # Look inside meta.spec.label_feature to identify the label_column if the user did not specify it
    if label_column is None:
        if is_feature_vector:
            label_column = df_artifact.meta.spec.label_feature.split(".")[1]
        else:
            raise ValueError("No label_column was given, please add a label_column.")

    # Use the feature vector as dataframe
    df = df_artifact.as_df()

    # Ensure k is not bigger than the total number of features
    if k > df.shape[1]:
        raise ValueError(
            f"K cannot be bigger than the total number of features ({df.shape[1]}). Please choose a smaller K."
        )
    elif k < 1:
        raise ValueError("K cannot be smaller than 1. Please choose a bigger K.")

    # Create a sample dataframe of the original feature vector
    if sample_ratio:
        df = (
            df.groupby(label_column)
            .apply(lambda x: x.sample(frac=sample_ratio))
            .reset_index(drop=True)
        )
        df = df.dropna()

    # Set feature vector and labels
    y = df.pop(label_column)
    X = df

    if np.object in list(X.dtypes) and ignore_type_errors is False:
        raise ValueError(
            f"{df.select_dtypes(include=['object']).columns.tolist()} are neither float or int."
        )

    # Create selected statistical estimators
    stat_functions_list = {
        stat_name: SelectKBest(
            score_func=create_class(f"sklearn.feature_selection.{stat_name}"), k=k
        )
        for stat_name in stat_filters
    }
    requires_abs = ["chi2"]

    # Run statistic filters
    selected_features_agg = {}
    stats_df = pd.DataFrame(index=X.columns).dropna()

    for stat_name, stat_func in stat_functions_list.items():
        try:
            params = (X, y) if stat_name in requires_abs else (abs(X), y)
            stat = stat_func.fit(*params)

            # Collect stat function results
            stat_df = pd.DataFrame(
                index=X.columns, columns=[stat_name], data=stat.scores_
            )
            plot_stat(context, stat_name, stat_df)
            stats_df = stats_df.join(stat_df)

            # Select K Best features
            selected_features = X.columns[stat_func.get_support()]
            selected_features_agg[stat_name] = selected_features

        except Exception as e:
            context.logger.info(f"Couldn't calculate {stat_name} because of: {e}")

    # Create models from class name / json file / json params
    all_sklearn_estimators = dict(all_estimators()) if len(model_filters) > 0 else {}
    selected_models = {}
    for model_name, model in model_filters.items():
        if ".json" in model:
            current_model = json.load(open(model, "r"))
            classifier_class = create_class(current_model["META"]["class"])
            selected_models[model_name] = classifier_class(**current_model["CLASS"])
        elif model in all_sklearn_estimators:
            selected_models[model_name] = all_sklearn_estimators[model_name]()

        else:
            try:
                current_model = json.loads(model)
                classifier_class = create_class(current_model["META"]["class"])
                selected_models[model_name] = classifier_class(**current_model["CLASS"])
            except Exception as e:
                context.logger.info(f"unable to load {model} because of: {e}")

    # Run model filters
    models_df = pd.DataFrame(index=X.columns)
    for model_name, model in selected_models.items():

        if model_name == "LogisticRegression":
            model.set_params(solver="liblinear")

        # Train model and get feature importance
        select_from_model = SelectFromModel(model).fit(X, y)
        feature_idx = select_from_model.get_support()
        feature_names = X.columns[feature_idx]
        selected_features_agg[model_name] = feature_names.tolist()

        # Collect model feature importance
        if hasattr(select_from_model.estimator_, "coef_"):
            stat_df = select_from_model.estimator_.coef_
        elif hasattr(select_from_model.estimator_, "feature_importances_"):
            stat_df = select_from_model.estimator_.feature_importances_

        stat_df = pd.DataFrame(index=X.columns, columns=[model_name], data=stat_df[0])
        models_df = models_df.join(stat_df)

        plot_stat(context, model_name, stat_df)

    # Create feature_scores DF with stat & model filters scores
    result_matrix_df = pd.concat([stats_df, models_df], axis=1, sort=False)
    context.log_dataset(
        key="feature_scores",
        df=result_matrix_df,
        local_path="feature_scores.parquet",
        format="parquet",
    )
    if max_scaled_scores:
        normalized_df = result_matrix_df.replace([np.inf, -np.inf], np.nan).values
        min_max_scaler = MinMaxScaler()
        normalized_df = min_max_scaler.fit_transform(normalized_df)
        normalized_df = pd.DataFrame(
            data=normalized_df,
            columns=result_matrix_df.columns,
            index=result_matrix_df.index,
        )
        context.log_dataset(
            key="max_scaled_scores_feature_scores",
            df=normalized_df,
            local_path="max_scaled_scores_feature_scores.parquet",
            format="parquet",
        )

    # Create feature count DataFrame
    for test_name in selected_features_agg:
        result_matrix_df[test_name] = [
            1 if x in selected_features_agg[test_name] else 0 for x in X.columns
        ]
    result_matrix_df.loc[:, "num_votes"] = result_matrix_df.sum(axis=1)
    context.log_dataset(
        key="selected_features_count",
        df=result_matrix_df,
        local_path="selected_features_count.parquet",
        format="parquet",
    )

    # How many votes are needed for a feature to be selected?
    if isinstance(min_votes, int):
        votes_needed = min_votes
    else:
        num_filters = len(stat_filters) + len(model_filters)
        votes_needed = int(np.floor(num_filters * max(min(min_votes, 1), 0)))
    context.logger.info(f"votes needed to be selected: {votes_needed}")

    # Create final feature dataframe
    selected_features = result_matrix_df[
        result_matrix_df.num_votes >= votes_needed
    ].index.tolist()
    good_feature_df = df.loc[:, selected_features]
    final_df = pd.concat([good_feature_df, y], axis=1)
    context.log_dataset(
        key="selected_features",
        df=final_df,
        local_path="selected_features.parquet",
        format="parquet",
    )

    # Creating a new feature vector containing only the identified top features
    if is_feature_vector and df_artifact.meta.spec.features and output_vector_name:
        # Selecting the top K features from our top feature dataframe
        selected_features = result_matrix_df.head(k).index

        # Match the selected feature names to the FS Feature annotations
        matched_selections = [
            feature
            for feature in list(df_artifact.meta.spec.features)
            for selected in list(selected_features)
            if feature.endswith(selected)
        ]

        # Defining our new feature vector
        top_features_fv = fs.FeatureVector(
            output_vector_name,
            matched_selections,
            label_feature="labels.label",
            description="feature vector composed strictly of our top features",
        )

        # Saving
        top_features_fv.save()
        fs.get_offline_features(top_features_fv, target=ParquetTarget())

        # Logging our new feature vector URI
        context.log_result("top_features_vector", top_features_fv.uri)
