# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from lightgbm import LGBMRegressor
import lightgbm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings("ignore")


'''
https://www.kaggle.com/kabure/simple-eda-lightgbm-autotuning-w-hyperopt
'''


answer_col = 'scalar_coupling_constant'
type_col = 'type'

def create_features(df):
    df['hop'] = df[type_col].apply(lambda x: x[0])

    return df


def metric(df, preds):
    ''' Defining the Metric to score our optimizer '''
    df['diff'] = (df[answer_col] - preds).abs()
    return np.log(df.groupby(type_col)['diff'].mean().map(lambda x: max(x, 1e-9))).mean()


def load_df():
    # df_pot_energy = pd.read_csv('./input/potential_energy.csv')
    # df_mul_charges = pd.read_csv('./input/mulliken_charges.csv')
    # df_scal_coup_contrib = pd.read_csv('./input/scalar_coupling_contributions.csv')
    # df_magn_shield_tensor = pd.read_csv('./input/magnetic_shielding_tensors.csv')
    # df_dipole_moment = pd.read_csv('./input/dipole_moments.csv')
    # df_structure = pd.read_csv('./input/structures.csv')
    df_train = pd.read_csv('./input/train.csv')
    df_test = pd.read_csv('./input/test.csv')

    df_train = create_features(df_train)
    df_test = create_features(df_test)

    return df_train, df_test


def label_encode(df_train, df_test, enc_cols):
    ''' 相関係数計算のため、文字列の特徴量をエンコード '''
    for col in df_train.columns:
        if col in enc_cols:
            lbl = LabelEncoder()
            lbl.fit(list(df_train[col].values) + list(df_test[col].values))
            df_train[col] = lbl.transform(list(df_train[col].values))
            df_test[col] = lbl.transform(list(df_test[col].values))

    return df_train, df_test


def select_isolate_cols(df_train, df_test):
    ''' 相関が高い特徴量を削除 '''
    # Threshold for removing correlated variables
    threshold = 0.95

    # Absolute value correlation matrix
    corr_matrix = df_train.corr().abs()

    # Getting the upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print('There are {} columns to remove. {}'.format(len(to_drop), tuple(to_drop)))

    df_train = df_train.drop(columns=to_drop)
    df_test = df_test.drop(columns=to_drop)

    print('Training shape: ', df_train.shape)
    print('Testing shape: ', df_test.shape)

    return df_train, df_test


def preprocess():
    ''' 前処理 '''
    df_train, df_test = load_df()

    enc_cols = ('hop', type_col)
    df_train, df_test = label_encode(df_train, df_test, enc_cols)
    df_train, df_test = select_isolate_cols(df_train, df_test)

    return df_train, df_test

def define_space():
    if False:
        hyper_space = {'objective': 'regression',
                       'metric':'mae',
                       'boosting':'gbdt',
                       #'n_estimators': hp.choice('n_estimators', [25, 40, 50, 75, 100, 250, 500]),
                       'max_depth':  hp.choice('max_depth', [5, 8, 10, 12, 15]),
                       'num_leaves': hp.choice('num_leaves', [100, 250, 500, 650, 750, 1000,1300]),
                       'subsample': hp.choice('subsample', [.3, .5, .7, .8, 1]),
                       'colsample_bytree': hp.choice('colsample_bytree', [ .6, .7, .8, .9, 1]),
                       'learning_rate': hp.choice('learning_rate', [.1, .2, .3]),
                       'reg_alpha': hp.choice('reg_alpha', [.1, .2, .3, .4, .5, .6]),
                       'reg_lambda':  hp.choice('reg_lambda', [.1, .2, .3, .4, .5, .6]),
                       'min_child_samples': hp.choice('min_child_samples', [20, 45, 70, 100])}
    else:
        hyper_space = {'objective': 'regression',
                       'metric':'mae',
                       'boosting':'gbdt',
                       'max_depth':  hp.choice('max_depth', [12]),
                       'num_leaves': hp.choice('num_leaves', [1300]),
                       'subsample': hp.choice('subsample', [.3]),
                       'colsample_bytree': hp.choice('colsample_bytree', [1]),
                       'learning_rate': hp.choice('learning_rate', [.1]),
                       'reg_alpha': hp.choice('reg_alpha', [.1]),
                       'reg_lambda':  hp.choice('reg_lambda', [.1]),
                       'min_child_samples': hp.choice('min_child_samples', [45])}

    return hyper_space


def main():
    df_train, df_test = preprocess()

    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(df_train.drop(answer_col, axis=1),
                                                      df_train[answer_col],
                                                      test_size = 0.10,
                                                      random_state = 0)

    # metric用
    df_val = pd.DataFrame({type_col: X_val[type_col]})
    df_val[answer_col] = y_val

    print(X_train.head())
    drop_cols = ['id', 'atom_index_0', 'atom_index_1', 'molecule_name']

    print("Traing features: {}".format(X_train.drop(drop_cols, axis=1).columns))
    X_train = X_train.drop(drop_cols, axis=1).values
    y_train = y_train.values

    X_val = X_val.drop(drop_cols, axis=1).values
    y_val = y_val.values

    X_test = df_test.drop(drop_cols, axis=1).values

    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Validation set has {} samples.".format(X_val.shape[0]))

    # Define searched space
    hyper_space = define_space()

    # metric用
    lgtrain = lightgbm.Dataset(X_train, label=y_train, params={'verbose': -1})
    lgval = lightgbm.Dataset(X_val, label=y_val, params={'verbose': -1})

    def evaluate_metric(params):
        model_lgb = lightgbm.train(params, lgtrain, 500,
                                   valid_sets=[lgtrain, lgval], early_stopping_rounds=20,
                                   verbose_eval=500)

        pred = model_lgb.predict(X_val)
        score = metric(df_val, pred)
        print('## metric: {}'.format(score))
        results = {
            'loss': score,
            'status': STATUS_OK,
            'stats_running': STATUS_RUNNING
        }
        return results


    def search_param(hyper_space):
        print('### Start search best params ###')
        # Trail
        trials = Trials()
        # Set algoritm parameters
        algo = partial(tpe.suggest, n_startup_jobs=-1)
        # Seting the number of Evals
        MAX_EVALS = 15
        # Fit Tree Parzen Estimator
        best_vals = fmin(evaluate_metric,
                         space=hyper_space,
                         verbose=-1,
                         algo=algo,
                         max_evals=MAX_EVALS,
                         trials=trials)

        # Print best parameters
        best_params = space_eval(hyper_space, best_vals)
        print("BEST PARAMETERS: " + str(best_params))
        return best_params


    search = True
    if search:
        best_params = search_param(hyper_space)
    else:
        best_params = {'boosting': 'gbdt', 'colsample_bytree': 1, 'learning_rate': 0.1,
                       'max_depth': 12, 'metric': 'mae', 'min_child_samples': 45,
                       'num_leaves': 1300, 'objective': 'regression',
                       'reg_alpha': 0.2, 'reg_lambda': 0.1, 'subsample': 0.3}

    best_params['verbose'] = -1

    print('### Start training ###')
    model_lgb = lightgbm.train(best_params, lgtrain, 4000,
                               valid_sets=[lgtrain, lgval], early_stopping_rounds=30,
                               verbose_eval=500)

    lgb_pred = model_lgb.predict(X_test)

    df_test[answer_col] = lgb_pred
    df_test[['id', answer_col]].to_csv("molecular_struct_sub.csv", index=False)


if __name__ == '__main__':
    main()
