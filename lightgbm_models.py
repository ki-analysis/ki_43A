# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:10:20 2021

@author: serge
"""
from constants import RANDOM_STATE, N_JOBS
import lightgbm as lgb


def make_lightgbm_model(
    learning_task,
    objective,
    metric=None,
    tree_learner='feature',  # best for small datasets
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
):
    # default objective
    if objective is None:
        if learning_task == 'binary':
            objective = 'binary'
        elif learning_task == 'multiclass':
            objective = 'multiclass'
        else:
            objective = 'mae'

    # default metric based on metric
    if metric is None:
        if objective == 'binary':
            metric = 'auc'
        elif objective == 'multiclass':
            metric = 'auc_mu'
        else:
            metric = 'mae'

    if learning_task in {'binary', 'multiclass'}:
        estimator = lgb.LGBMClassifier(
            objective=objective,
            metric=metric,
            tree_learner=tree_learner,
            random_state=RANDOM_STATE,
            silent=True,
            verbose=-1,
            n_jobs=n_jobs,
        )
    else:
        estimator = lgb.LGBMRegressor(
            objective=objective,
            metric=metric,
            tree_learner=tree_learner,
            random_state=RANDOM_STATE,
            silent=True,
            verbose=-1,
            n_jobs=n_jobs,
        )
    return estimator
