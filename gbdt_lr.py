'''
lightgbm man page: 
http://lightgbm.readthedocs.io/en/latest/python/lightgbm.html#

reference(gbdt+lr example):
https://github.com/neal668/LightGBM-GBDT-LR/blob/master/GBFT%2BLR_simple.py
'''



from sklearn.datasets import load_iris
import numpy as np

import lightgbm as lgb
import pandas as pd

## build data
iris=load_iris()
iris = pd.DataFrame(load_iris().data)
iris.columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
iris['Species'] = load_iris().target

## lable binary 
iris.Species=iris.Species%2


## train test split
train=iris[0:120]
test=iris[120:]
X_train=train.filter(items=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
X_test=test.filter(items=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
y_train=train[[train.Species.name]]
y_test=test[[test.Species.name]]



## build lgb dataset

#########################################################
##label need reshape from (shape[0],1)  to (shape[0],)###
#########################################################
#lgb_train = lgb.Dataset(X_train.as_matrix(), y_train.values.reshape(y_train.shape[0],),feature_name=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
lgb_train = lgb.Dataset(X_train.as_matrix(), y_train.values.reshape(y_train.shape[0],))
lgb_eval = lgb.Dataset(X_test.as_matrix(), y_test.values.reshape(y_test.shape[0],), reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 63,
    'num_trees': 100,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# number of leaves,will be used in feature transformation
num_leaf = 63


gbm = lgb.train(params=params,train_set=lgb_train,num_boost_round=3000,valid_sets=lgb_train)

y_pred = gbm.predict(X_train,pred_leaf=True)

## build train matrix
transformed_training_matrix = np.zeros([len(y_pred),len(y_pred[0]) * num_leaf],dtype=np.int64)
for i in range(0,len(y_pred)):
	temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
	transformed_training_matrix[i][temp] += 1




y_pred = gbm.predict(X_test,pred_leaf=True)

## build test matrix
transformed_testing_matrix = np.zeros([len(y_pred),len(y_pred[0]) * num_leaf],dtype=np.int64)
for i in range(0,len(y_pred)):
	temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
	transformed_testing_matrix[i][temp] += 1



print('Feature importances:', list(gbm.feature_importance()))
print('Feature importances:', list(gbm.feature_importance("gain")))




c = np.array([1,0.5,0.1,0.05,0.01,0.005,0.001])
for t in range(0,len(c)):
    lm = LogisticRegression(penalty='l2',C=c[t]) # logestic model construction
    lm.fit(transformed_training_matrix,y_train.values.reshape(y_train.shape[0],))  # fitting the data
    y_pred_est = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label


