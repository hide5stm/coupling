import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')
df_gender_submission = pd.read_csv('./input/gender_submission.csv')

# Sexを0, 1に変換
genders = {'male': 0, 'female': 1}
df_train['Sex'] = df_train['Sex'].map(genders)
df_test['Sex'] = df_test['Sex'].map(genders)

# ダミー変数化
df_train = pd.get_dummies(df_train, columns=['Embarked'])
df_test = pd.get_dummies(df_test, columns=['Embarked'])

# 不要カラムの削除
drops = ['PassengerId', 'Name', 'Cabin', 'Ticket']
df_train.drop(drops, axis=1, inplace=True)
gx_test = df_test.drop(drops, axis=1).copy()


print(df_train.head(10))

x_train = df_train.iloc[:,1:]
y_train = df_train['Survived']

# 交差検証
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
    # learning
    gbm.fit(x_cv_train, y_cv_train,
            eval_set=[(x_cv_test, y_cv_test)],
            early_stopping_rounds=10)

    y_pred = gbm.predict(x_cv_test, num_iteration=gbm.best_iteration_)
    print(round(accuracy_score(y_cv_test, y_pred) * 100, 2))


    
gbm = lgb.LGBMClassifier(**params)
# learning
gbm.fit(x_cv_train, y_cv_train,
        eval_set=[(x_train, y_train)],
        early_stopping_rounds=10)

y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)

submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': y_pred
})

submission.to_csv('submission.csv', index=False)
