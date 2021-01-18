# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:40:03 2021

@author: serge
"""
import pickle
import pandas as pd

# import data from excel files
df = pd.read_excel("data/DataRequest_Sanchez-Alonso_12.10.20.xlsx", na_values=[99, '99', 999, '999', 9999, '9999'])

# import train/test split
test_train = pd.read_excel("data/BEAN_testing_training_n130.xlsx")

# this entire exercise will not touch the test data, so all we need is the training data
train_ids = test_train.loc[test_train['Dataset'] == 'Training', 'ID']
df = df[df['FSID'].isin(train_ids)]

# the TENSION scale variables
tension_covariates = [
    "feelsad",
    "noapptit",
    "mmryloss",
    "frustrat",
    "insomnia",
    "coldbody",
    "runaway",
    "headache",
    "afraid",
    "feeltird",
    "helpless",
    "shakines",
    "sexprob",
    "priodprb",
    "bealone",
    "painbody",
    "homesick",
    "feelhot",
    "vagdisch",
    "feeldizz",
    "hartpalp",
    "losscntr",
    "brethles",
    "hairwhit",
    "wtchnge",
]

'''
AGEM_AN03 has the actual age of the child for the measurement stored in HAZ_AN01
we have to exclude HAZ values that happened after the MULLEN scores at 6m and 24m

for predicting 6m mullen, can use: HAZ_AN01 through HAZ_AN03 (HAZ_AN04 can happen at 6m or after)
for predicing 24m mullen, can use: HAZ_AN01 through HAZ_AN09 (HAZ_AN10 can happen at 24m or after)
caveat: HAZ_AN07 has missing values, so we're excluding that one
'''
haz_covariates_for_6m = [f'HAZ_AN0{i}' for i in [1, 2, 3]]
haz_covariates_for_24m = [f'HAZ_AN0{i}' for i in [1, 2, 3, 4, 5, 6, 8, 9]]

other_covariates = [
    'wall_1',  # categorical but only 2 unique values (3 and 4) so doesn't need special codeing
    'medu_1',
    'fedu_1',
    'inco_1',
    'SEX',
    'room_1',
    'WT_1',
    'HT_1',
    'MUAC_1',
]

output_covariates_6m = [
    "gmraw_6",
    "gmtsc_6",
    "vrraw_6",
    "vrtsc_6",
    "fmraw_6",
    "fmtsc_6",
    "rlraw_6",
    "rltsc_6",
    "elraw_6",
    "eltsc_6",
]

output_covariates_24m = [
    "gmraw_24",
    "gmtsc_24",
    "vrraw_24",
    "vrtsc_24",
    "fmraw_24",
    "fmtsc_24",
    "rlraw_24",
    "rltsc_24",
    "elraw_24",
    "eltsc_24",
]


'''
now we can put together all the covariates for 6m and 24m and save for easy access
'''
input_covariates_6m = tension_covariates + haz_covariates_for_6m + other_covariates
output_covariates_6m_raw = [i for i in output_covariates_6m if 'raw' in i]
df['mullen_6m_raw_average'] = df[output_covariates_6m_raw].mean(1)  # simple average
output_covariates_6m_tsc = [i for i in output_covariates_6m if 'tsc' in i]
df['mullen_6m_tsc_average'] = df[output_covariates_6m_tsc].mean(1)  # simple average
df_6m = df[input_covariates_6m + ['FSID', 'mullen_6m_raw_average', 'mullen_6m_tsc_average']]

input_covariates_24m = tension_covariates + haz_covariates_for_24m + other_covariates
output_covariates_24m_raw = [i for i in output_covariates_24m if 'raw' in i]
df['mullen_24m_raw_average'] = df[output_covariates_24m_raw].mean(1)  # simple average
output_covariates_24m_tsc = [i for i in output_covariates_24m if 'tsc' in i]
df['mullen_24m_tsc_average'] = df[output_covariates_24m_tsc].mean(1)  # simple average
df_24m = df[input_covariates_24m + ['FSID', 'mullen_24m_raw_average', 'mullen_24m_tsc_average']]
# not all mullen scores are available for 24m
df_24m = df_24m[~df_24m['mullen_24m_raw_average'].isnull()]

# save to disk
with open('data/processed_data.pickle', 'wb') as f:
    pickle.dump((df_6m, input_covariates_6m, df_24m, input_covariates_24m), f)
