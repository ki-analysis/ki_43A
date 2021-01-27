# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:21:39 2021

@author: sergey feldman
"""
from hyperopt import hp
from hyperopt.pyll import scope

N_JOBS = 8
RANDOM_STATE = 42
N_SPLITS = 4  # for both inner and outer
N_HYPEROPT_EVALS = 50  # number of lightgbm hyperopt evaluations
N_HYPEROPT_RANDOM_START = 20  # hyperopt starts with this many random evaluations
# autogluon is an automated ML ensemble. this variable defines how long it runs for. more is better
AUTOGLUON_N_SEC = 60 * 5

# the lightgbm model has so many hyperparameters
# that a grid search is not feasible.
# we instead do a smart search over the space of hyperparameters
# as defined below
HYPEROPT_LIGHTGBM_SPACE = {
    "n_estimators": hp.choice("n_estimators", [100, 250, 500, 1000, 2000]),
    "learning_rate": hp.choice("learning_rate", [0.1, 0.05, 0.01, 0.005, 0.001]),
    "num_leaves": scope.int(2 ** hp.quniform("num_leaves", 2, 7, 1)),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.4, 1, 0.1),
    "subsample": hp.quniform("subsample", 0.4, 1, 0.1),
    "min_child_samples": scope.int(2 ** hp.quniform("min_child_samples", 0, 7, 1)),
    "min_child_weight": 10 ** hp.quniform("min_child_weight", -6, 0, 1),
    "reg_alpha": hp.choice("reg_alpha", [0, 10 ** hp.quniform("reg_alpha_pos", -6, 1, 1)]),
    "reg_lambda": hp.choice("reg_lambda", [0, 10 ** hp.quniform("reg_lambda_pos", -6, 1, 1)]),
    "max_depth": scope.int(hp.choice("max_depth", [-1, 2 ** hp.quniform("max_depth_pos", 1, 4, 1)])),
}
