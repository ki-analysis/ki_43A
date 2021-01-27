# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:44:10 2021

@author: serge
"""

from constants import RANDOM_STATE
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


"""
Classification models
"""

# Linear SVC classifier
svc_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "svc",
            SVC(kernel="linear", class_weight="balanced", probability=True, tol=1e-4, random_state=RANDOM_STATE),
        ),
    ]
)
svc_grid = {
    "svc__C": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
}

# logistic regression classifier
lr_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "logistic",
            LogisticRegression(
                solver="saga", max_iter=10000, penalty="elasticnet", l1_ratio=0.1, random_state=RANDOM_STATE
            ),  # without elasticnet penalty, LR can get awful performance
        ),
    ]
)
lr_grid = {
    "logistic__C": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
}


"""
Regression models
"""

# ridge regression
ridge_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "ridge",
            Ridge(tol=1e-4, random_state=RANDOM_STATE),
        ),
    ]
)
ridge_grid = {
    "ridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
}

# elastic net
elasticnet_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        (
            "elasticnet",
            ElasticNet(max_iter=10000, l1_ratio=0.1, random_state=RANDOM_STATE),
        ),
    ]
)
elasticnet_grid = {
    "elasticnet__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
}

# Linear SVR regressor
svr_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("svr", SVR(kernel="linear", tol=1e-4)),
    ]
)
svr_grid = {
    "svr__C": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
}
