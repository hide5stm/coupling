import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def load_df():
    df_train = pd.read_csv('./input/train.csv')
    df_test = pd.read_csv('./input/test.csv')
    df_sample_submission = pd.read_csv('./input/sample_submission.csv')


def main():
    # 不要カラムの削除
    drops = ['PassengerId', 'Name', 'Cabin', 'Ticket']
    df_train.drop(drops, axis=1, inplace=True)
    df_test.drop(drops, axis=1, inplace=True)

    # id,molecule_name,atom_index_0,atom_index_1,type,scalar_coupling_constant
    

    print(df_train.head(10))

    

x_train = df_train.iloc[:,1:]
y_train = df_train['Survived']

kf = KFold(n_splits=3)

params = {
    'objective': 'binary',
    'learning_rate': 0.1,
    'num_leaves': 300
}

for train_index, test_index in kf.split(x_train, y_train):
    x_cv_train = x_train.iloc[train_index]
    x_cv_test = x_train.iloc[train_index]
    y_cv_train = y_train.iloc[train_index]
    y_cv_test = y_train.iloc[train_index]

    gbm = lgb.LGBMClassifier(**params)
    # learningl
    gbm.fit(x_cv_train, y_cv_train,
            eval_set=[(x_cv_test, y_cv_test)],
            early_stopping_rounds=10)

    y_pred = gbm.predict(x_cv_test, num_iteration=gbm.best_iteration_)
    print(round(accuracy_score(y_cv_test, y_pred) * 100, 2))
    pd.DataFrame(y_pred).to_csv('y_pred.csv')


