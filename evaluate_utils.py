# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:31:42 2021

@author: sergey feldman
"""
from splitting_utils import get_cv
from lightgbm_models import make_lightgbm_model
from constants import N_JOBS, RANDOM_STATE, N_SPLITS, N_HYPEROPT_EVALS
from constants import HYPEROPT_LIGHTGBM_SPACE, N_HYPEROPT_RANDOM_START, AUTOGLUON_N_SEC
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from sklearn.metrics import get_scorer
from hyperopt import fmin, tpe, Trials, space_eval
from autogluon import TabularPrediction as task


def get_default_scoring(learning_task):
    # this function defines the "default" scoring
    # that will be used to evaluate ML models
    if learning_task == "binary":
        return "roc_auc"
    elif learning_task == "multiclass":
        return "roc_auc_ovr_weighted"
    else:
        return "neg_mean_absolute_error"


def sklearn_pipeline_evaluator(
    X,
    y,
    pipeline,
    param_grid,
    groups=None,
    outer_cv=None,
    learning_task="regression",
    scoring=None,
    random_state=RANDOM_STATE,
    n_splits=N_SPLITS,
    n_jobs=N_JOBS,
):
    if scoring == None:
        scoring = get_default_scoring(learning_task)

    # see here for learning metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
    inner_cv = get_cv(learning_task, groups is not None, n_splits, random_state)
    if outer_cv is None:
        outer_cv = get_cv(learning_task, groups is not None, n_splits, random_state)

    # sklearn models can be optimized via grid search because they have few hyperparams
    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring=scoring,
        n_jobs=N_JOBS,
        refit=True,
    )

    # this does nested cross-validation for you!
    cv_results = cross_validate(
        clf, X=X, y=y, groups=groups, cv=outer_cv, scoring=scoring, return_estimator=True, n_jobs=n_jobs
    )
    return cv_results


def lightgbm_hyperopt_evaluator(
    X,
    y,
    groups=None,
    outer_cv=None,
    param_space=HYPEROPT_LIGHTGBM_SPACE,
    learning_task="regression",
    scoring=None,
    lightgbm_objective=None,
    lightgbm_metric=None,
    random_state=RANDOM_STATE,
    n_splits=N_SPLITS,
    n_jobs=N_JOBS,
):

    # get scoring and CV set up
    if scoring == None:
        scoring = get_default_scoring(learning_task)

    scorer = get_scorer(scoring)  # a function that does the scoring

    if outer_cv is None:
        outer_cv = get_cv(learning_task, groups is not None, n_splits, random_state)
        if groups is None:
            outer_cv = outer_cv.split(X, y)
        else:
            outer_cv = outer_cv.split(X, y, groups=groups)

    # run the loop over the outer folds
    # we can't use sklearn's nice nested approach unfortunately
    nested_scores = []
    estimators = []
    for train_inds, test_inds in outer_cv:
        # define the lightgbm model
        lgb_model = make_lightgbm_model(
            learning_task,
            objective=lightgbm_objective,
            metric=lightgbm_metric,
            tree_learner="feature",  # best for small datasets
            random_state=random_state,
            n_jobs=n_jobs,
        )
        # split the data
        X_train, y_train = X[train_inds, :], y[train_inds]
        X_test, y_test = X[test_inds, :], y[test_inds]
        if groups is not None:
            groups_train = groups[train_inds]
        else:
            groups_train = None

        # this is the objective function that we have to minimize with hyperopt's fmin function
        def obj(params):
            # set the parametrs for the lightgbm model
            lgb_model.set_params(**params)
            # inner cross-validation
            inner_cv = get_cv(learning_task, groups is not None, n_splits, random_state)
            # get the inner cross-validation scores for this set of params
            scores = cross_val_score(
                lgb_model, X=X_train, y=y_train, groups=groups_train, cv=inner_cv, scoring=scoring, n_jobs=N_JOBS
            )
            # minimize means we negative mean absolute error at the end
            return -np.mean(scores)

        # this is how hyperopt works. you need to give it an objective and a parameter space
        trials = Trials()
        _ = fmin(
            fn=obj,
            space=param_space,
            algo=partial(tpe.suggest, n_startup_jobs=N_HYPEROPT_RANDOM_START),
            max_evals=N_HYPEROPT_EVALS,
            trials=trials,
            rstate=np.random.RandomState(random_state),
            show_progressbar=False,
            verbose=False,
        )

        # these are the best parameters hyperopt found
        # hyperopt has some warts (problems with hp.choice) so we need to do this:
        best_params = space_eval(param_space, trials.argmin)

        # we set the lgb_model to have these parameters and train on the entire training set
        lgb_model.set_params(**best_params)
        lgb_model.fit(X_train, y_train)

        # now we can evaluate on the test
        score = scorer(lgb_model, X_test, y_test)

        # store everything
        nested_scores.append(score)
        estimators.append(lgb_model)

    return {"test_score": np.array(nested_scores), "estimator": estimators}


def autogluon_evaluator(
    X,
    y,
    groups=None,
    outer_cv=None,
    learning_task="regression",
    scoring=None,
    autogluon_eval_metric=None,
    random_state=RANDOM_STATE,
    n_splits=N_SPLITS,
    n_jobs=N_JOBS,
):
    # get scoring and CV set up
    if scoring == None:
        scoring = get_default_scoring(learning_task)

    scorer = get_scorer(scoring)

    if outer_cv is None:
        outer_cv = get_cv(learning_task, groups is not None, n_splits, random_state)
        if groups is None:
            outer_cv = outer_cv.split(X, y)
        else:
            outer_cv = outer_cv.split(X, y, groups=groups)

    # autogluon eval metrics have different names
    if autogluon_eval_metric is None:
        if learning_task == "binary":
            autogluon_eval_metric = "roc_auc"
        elif learning_task == "multiclass":
            # no multiclass roc_auc available
            autogluon_eval_metric = "f1_weighted"
        else:
            autogluon_eval_metric = "mean_absolute_error"

    # autogluon wants pandas dataframes
    data_df = pd.DataFrame(X)
    data_df["y"] = y

    # run the loop over the outer folds
    nested_scores = []
    estimators = []
    for train_inds, test_inds in outer_cv:
        # define train and test splits
        data_df_train = data_df.iloc[train_inds, :]
        data_df_test = data_df.iloc[test_inds, :]

        # define the autogluon model
        # there is no inner CV for autogluon
        # it does its own inner optimization
        # for as long as you let it (set with time_limits)
        autogluon_model = task.fit(
            data_df_train,
            "y",
            time_limits=AUTOGLUON_N_SEC,
            presets="best_quality",
            eval_metric=autogluon_eval_metric,
            problem_type=learning_task,
            verbosity=0,
        )

        # now we can evalute the autogluon model on the test set
        score = scorer(autogluon_model, data_df_test, data_df_test["y"])

        # store everything
        estimators.append(autogluon_model)
        nested_scores.append(score)

    return {"test_score": np.array(nested_scores), "estimator": estimators}
