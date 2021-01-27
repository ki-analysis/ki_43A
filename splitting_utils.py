# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:18:37 2021

@author: serge
"""

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit
from constants import RANDOM_STATE, N_SPLITS


def get_cv(learning_task, groups=False, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    # this function creates the appropriate cross-validation splitter.
    # if a group is provided, it is used to stratify the splits
    # otherwise if it's a classification problem, the classes as used as the strata.
    # for regression there is no stratification
    # there is some wonkiness when you have exactly one split - that's a scikit learn issue
    assert n_splits >= 1
    if n_splits > 1:
        if groups:
            cv = GroupKFold(n_splits)
        elif learning_task in {"binary", "multiclass"}:
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits, shuffle=True, random_state=random_state)
    elif n_splits == 1:
        if groups:
            cv = GroupShuffleSplit(n_splits=1, train_size=0.75, test_size=0.25, random_state=random_state)
        elif learning_task in {"binary", "multiclass"}:
            cv = StratifiedShuffleSplit(n_splits=1, train_size=0.75, test_size=0.25, random_state=random_state)
        else:
            cv = ShuffleSplit(n_splits=1, train_size=0.75, test_size=0.25, random_state=random_state)
    return cv
