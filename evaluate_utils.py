# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:31:42 2021

@author: serge
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
#from autogluon import TabularPrediction as task


def get_default_scoring(learning_task):
    if learning_task == 'binary':
        return "roc_auc"
    elif learning_task == 'multiclass':
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
    learning_task='regression',
    scoring=None,
    random_state=RANDOM_STATE,
    n_splits=N_SPLITS,
    n_jobs=N_JOBS
):
    if scoring == None:
        scoring = get_default_scoring(learning_task)

    # see here for learning metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
    inner_cv = get_cv(learning_task, groups is not None, n_splits, random_state)
    if outer_cv is None:
        outer_cv = get_cv(learning_task, groups is not None, n_splits, random_state)
    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring=scoring,
        n_jobs=N_JOBS,
        refit=True,

    )
    cv_results = cross_validate(clf, X=X, y=y, groups=groups, cv=outer_cv,
                                scoring=scoring, return_estimator=True, n_jobs=n_jobs)
    return cv_results


def lightgbm_hyperopt_evaluator(
    X,
    y,
    groups=None,
    outer_cv=None,
    param_space=HYPEROPT_LIGHTGBM_SPACE,
    learning_task='regression',
    scoring=None,
    lightgbm_objective=None,
    lightgbm_metric=None,
    random_state=RANDOM_STATE,
    n_splits=N_SPLITS,
    n_jobs=N_JOBS
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

    # run the loop over the outer folds
    nested_scores = []
    estimators = []
    for train_inds, test_inds in outer_cv:
        lgb_model = make_lightgbm_model(
            learning_task,
            objective=lightgbm_objective,
            metric=lightgbm_metric,
            tree_learner='feature',  # best for small datasets
            random_state=random_state,
            n_jobs=n_jobs
        )
        X_train, y_train = X[train_inds, :], y[train_inds]
        X_test, y_test = X[test_inds, :], y[test_inds]
        if groups is not None:
            groups_train = groups[train_inds]
        else:
            groups_train = None

        def obj(params):
            lgb_model.set_params(**params)
            inner_cv = get_cv(learning_task, groups is not None, n_splits, random_state)
            scores = cross_val_score(
                lgb_model, X=X_train, y=y_train, groups=groups_train, cv=inner_cv, scoring=scoring, n_jobs=N_JOBS
            )
            return -np.mean(scores)

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
        # hyperopt has some problems with hp.choice so we need to do this:
        best_params = space_eval(param_space, trials.argmin)
        lgb_model.set_params(**best_params)
        lgb_model.fit(X_train, y_train)

        score = scorer(lgb_model, X_test, y_test)
        nested_scores.append(score)
        estimators.append(lgb_model)

    return {'test_score': np.array(nested_scores), 'estimator': estimators}


def autogluon_evaluator(
    X,
    y,
    groups=None,
    outer_cv=None,
    learning_task='regression',
    scoring=None,
    autogluon_eval_metric=None,
    random_state=RANDOM_STATE,
    n_splits=N_SPLITS,
    n_jobs=N_JOBS
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
        if learning_task == 'binary':
            autogluon_eval_metric = 'roc_auc'
        elif learning_task == 'multiclass':
            # no multiclass roc_auc!
            autogluon_eval_metric = 'f1_weighted'
        else:
            autogluon_eval_metric = 'mean_absolute_error'

    # autogluon wants dataframes
    data_df = pd.DataFrame(X)
    data_df["y"] = y

    # run the loop over the outer folds
    nested_scores = []
    estimators = []
    for train_inds, test_inds in outer_cv:
        data_df_train = data_df.iloc[train_inds, :]
        data_df_test = data_df.iloc[test_inds, :]
        predictor = task.fit(
            data_df_train,
            "y",
            time_limits=AUTOGLUON_N_SEC,
            presets="best_quality",
            eval_metric=autogluon_eval_metric,
            problem_type=learning_task,
            verbosity=0,
        )
        score = scorer(predictor, data_df_test, data_df.loc[test_inds, 'y'])
        estimators.append(predictor)
        nested_scores.append(score)

    return {'test_score': np.array(nested_scores), 'estimator': estimators}
