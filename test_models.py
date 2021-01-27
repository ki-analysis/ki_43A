# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:59:59 2021

@author: serge
"""
import numpy as np
from evaluate_utils import sklearn_pipeline_evaluator, lightgbm_hyperopt_evaluator
from linear_models import ridge_pipeline, ridge_grid
from linear_models import elasticnet_pipeline, elasticnet_grid
from linear_models import svr_pipeline, svr_grid
from sklearn.datasets import load_diabetes

# feel free to ignore all this. just making sure nothing crashes.

X, y = load_diabetes(return_X_y=True)

ridge_results = sklearn_pipeline_evaluator(
    X, y, ridge_pipeline, ridge_grid, groups=None, learning_task="regression", scoring="neg_mean_absolute_error"
)

elasticnet_results = sklearn_pipeline_evaluator(
    X,
    y,
    elasticnet_pipeline,
    elasticnet_grid,
    groups=None,
    learning_task="regression",
    scoring="neg_mean_absolute_error",
)

svr_results = sklearn_pipeline_evaluator(
    X, y, svr_pipeline, svr_grid, groups=None, learning_task="regression", scoring="neg_mean_absolute_error"
)

lightgbm_results = lightgbm_hyperopt_evaluator(
    X,
    y,
    groups=None,
    learning_task="regression",
    scoring="neg_mean_absolute_error",
    lightgbm_objective="mae",
    lightgbm_metric="mae",
)


for i in [ridge_results, elasticnet_results, svr_results, lightgbm_results]:
    print(-np.mean(i["test_score"]), np.std(i["test_score"]))
