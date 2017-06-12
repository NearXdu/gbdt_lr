'''
Gen csr_martrix as input feature for lr
'''
import zipfile
import pandas as pd
import numpy as np
from scipy import sparse
import scipy as sp
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from scipy.sparse import csr_matrix

'''
save sp.sparse.csr_matrix
'''
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

'''
load sp.sparse.csr_matrix
'''
def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

# 16 - 23 as validation
dfVal     = pd.read_csv("../final/train_16_23.csv")
dfAd      = pd.read_csv("../final/ad.csv")
dfUser    = pd.read_csv("../final/user.csv")
dfAppc    = pd.read_csv("../final/app_categories.csv")
dfPos     = pd.read_csv("../final/position.csv")

# 24 - 30 as train
dfTrain   = pd.read_csv("../final/train_24_30.csv")
dfTest    = pd.read_csv("../final/test.csv") 


### merge original feature
dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
dfVal   = pd.merge(dfVal,  dfAd, on="creativeID")
dfTest  = pd.merge(dfTest,  dfAd, on="creativeID")

dfTrain = pd.merge(dfTrain, dfUser, on="userID")
dfVal   = pd.merge(dfVal, dfUser, on="userID")
dfTest  = pd.merge(dfTest, dfUser, on="userID")

dfTrain = pd.merge(dfTrain, dfAppc, on="appID")
dfVal   = pd.merge(dfVal, dfAppc, on="appID")
dfTest  = pd.merge(dfTest, dfAppc, on="appID")

dfTrain = pd.merge(dfTrain, dfPos, on="positionID")
dfVal   = pd.merge(dfVal, dfPos, on="positionID")
dfTest  = pd.merge(dfTest, dfPos, on="positionID")

### prepare original features
feats_ad       = ['creativeID', 'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform']
feats_user     = ['age','gender','education','marriageStatus','haveBaby','hometown', 'residence']
feats_position = ['sitesetID', 'positionType','positionID', 'connectionType', 'telecomsOperator']
feats = []
feats = feats + feats_ad + feats_user + feats_position 
len(feats)



### prepare X & y
y_train = dfTrain["label"].values
X_train=dfTrain.filter(items=feats)



### prepare lightgbm dataset
lgb_train = lgb.Dataset(X_train.as_matrix(), y_train)

### prepare lightgbm param
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

### number of leaves,will be used in feature transformation
num_leaf = 63


### train models  ======>  num_boost_round decide number of trees for one iterator one tree built
gbm = lgb.train(params=params, train_set=lgb_train, num_boost_round = 100, valid_sets=lgb_train)

### save models
gbm.save_model("model.txt")

### predict leaf 
y_pred_train  = gbm.predict(X_train.as_matrix(), pred_leaf=True,num_iteration=gbm.best_iteration)

### build sp.sparse.csr_matrix for train set   =====>   input X for LR
transformed_training_matrix=csr_matrix((len(y_pred_train), len(y_pred_train[0])*num_leaf), dtype=np.int8)

### it may takes a long time
for i in range(0,len(y_pred_train)):
    temp = np.arange(len(y_pred_train[0])) * num_leaf - 1 + np.array(y_pred_train[i])
    transformed_training_matrix[i,temp] = 1


### save matrix   ======>   transformed_training_matrix.txt.npz
save_sparse_csr('transformed_training_matrix.txt',transformed_training_matrix)


### load matrix
# LR_X_train=load_sparse_csr('transformed_training_matrix.txt.npz')




