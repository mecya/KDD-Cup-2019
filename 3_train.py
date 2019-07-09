# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:06:57 2019

@author: zhangli
"""

#工具包导入

import numpy as np
import pandas as pd
import lightgbm as lgb
import os
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import json 
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from itertools import product
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
# 数据读取
path = '../data_set_phase2/beijing/' #or shanghai or shenguang
data=pd.read_csv(path+'beijing_features.csv',parse_dates=['req_time'])


########################################################################################
plan_mode_eta=['plan_model_'+str(i)+'_eta'  for i in range(1,12)]
pid_mode_feature= ['pid_mode_'+str(i) for i in range(1,12)]

feature_to_remove= ['req_time','click_mode','sid','pid','o','d']+ plan_mode_eta + pid_mode_feature
feature = [col for col in data.columns if col not in feature_to_remove]


########################################  特征选择  #############################################
feature = [col for col in feature if col not in ['req_time','click_mode','sid','pid','o','d']]
train_index = (data.req_time < '2018-12-01')
train = data[train_index][feature].reset_index(drop=True)
corr_matrix = train.corr().abs()
print(corr_matrix.head())
plt.figure(figsize=(9,9))
sns.heatmap(corr_matrix,annot= False, cmap='Blues')
plt.show()

threshold= 0.9

upper= corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))

to_drop= [column for column in upper.columns if any(upper[column]>threshold)]
print('There are %d columns to remove.' %(len(to_drop)))
###########################################################################################
#模型训练&验证
#评估指标设计
def f1_weighted(labels,preds):
    preds = np.argmax(preds.reshape(12, -1), axis=0)
    #print('labels:',labels)
    #print('preds:',preds)    
    score = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_weighted', score, True


def pred_proba(proba):
    res = []
#    pred_proba.tolist()
    for i, e in enumerate(proba):
        if e[3] >= 0.180: #0.2
            e[3] = 1
        if e[4] >= 0.118: #0.135
            e[4] = 1
        if e[6] >= 0.217:
            e[6] = 1
        if e[8] >= 0.21:
            e[8] = 1
        if e[9] >= 0.36:
            e[9] = 1
        if e[10] >= 0.285:
            e[10]=1
        if e[11] >= 0.34:
            e[11]=1
        res.append(e)
    df = pd.DataFrame(res)
    pred = df.idxmax(axis = 1)
#    pred = 
#    return pred
    return pred


###########################################训练#########################################
#feature = [col for col in feature if col not in to_drop]
train_index = (data.req_time < '2018-11-23') 
train_x     = data[train_index][feature].reset_index(drop=True)
train_y     = data[train_index].click_mode.reset_index(drop=True)

valid_index = (data.req_time > '2018-11-23') & (data.req_time < '2018-12-01')
valid_x     = data[valid_index][feature].reset_index(drop=True)
valid_y     = data[valid_index].click_mode.reset_index(drop=True)

test_index = (data.req_time > '2018-12-01')
test_x     = data[test_index][feature].reset_index(drop=True)

print(len(feature), feature)
lgb_model = lgb.LGBMClassifier(boosting_type="gbdt", num_leaves=100, reg_alpha=0, reg_lambda=0,
    max_depth=-1, n_estimators=2000, objective='multiclass',
    subsample=0.8, colsample_bytree=0.8, subsample_freq=1,min_child_samples = 50,  learning_rate=0.05, random_state=35, metric="None",n_jobs=-1)
eval_set = [(valid_x, valid_y)]
lgb_model.fit(train_x, train_y, eval_set=eval_set, eval_metric=f1_weighted, verbose=10, early_stopping_rounds=200)


lgb_model_1 = lgb.LGBMClassifier(boosting_type="gbdt", num_leaves=127, reg_alpha=0, reg_lambda=10,
                                max_depth=-1, n_estimators=2000, objective='multiclass',
                                subsample=0.886, colsample_bytree=0.886, subsample_freq=1,min_child_samples = 50,
                                learning_rate=0.05, random_state=615, metric="None",n_jobs=-1)

lgb_model_1.fit(train_x, train_y, eval_set=eval_set, eval_metric=f1_weighted, verbose=10, early_stopping_rounds=200)


#特征重要性分析

imp = pd.DataFrame()
imp['fea'] = feature
imp['imp'] = lgb_model.feature_importances_ 
imp = imp.sort_values('imp',ascending = False)
imp
plt.figure(figsize=[20,10])
sns.barplot(x = 'imp', y ='fea',data = imp.head(20))


#预测结果分析

proba = lgb_model.predict_proba(valid_x)
pred = pred_proba(proba)
score=f1_score(valid_y, pred, average='weighted')
print(score)
df_analysis = pd.DataFrame()
df_analysis['sid']   = data[valid_index]['sid']
df_analysis['label'] = valid_y.values
df_analysis['pred']  = pred
df_analysis['label'] = df_analysis['label'].astype(int)

score_df = pd.DataFrame(
    columns=['class_id', 'counts*f1_score', 'f1_score', 'precision', 'recall'])



#准确率、召回率分析

from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,recall_score,precision_score
dic_ = df_analysis['label'].value_counts(normalize = True)
def get_weighted_fscore(y_pred, y_true):
    f_score = 0
    for i in range(12):
        yt = y_true == i
        yp = y_pred == i
        f_score += dic_[i] * f1_score(y_true=yt, y_pred= yp)
        print(i,dic_[i],f1_score(y_true=yt, y_pred= yp), precision_score(y_true=yt, y_pred= yp),recall_score(y_true=yt, y_pred= yp))
    print(f_score)
score_df = get_weighted_fscore(y_true =df_analysis['label'] , y_pred = df_analysis['pred'])


#模型训练&提交

all_train_x              = data[data.req_time < '2018-12-01'][feature].reset_index(drop=True)
all_train_y              = data[data.req_time < '2018-12-01'].click_mode.reset_index(drop=True)
print(lgb_model.best_iteration_)
lgb_model.n_estimators   = lgb_model.best_iteration_
lgb_model.fit(all_train_x, all_train_y)
print('fit over')
result                   = pd.DataFrame()
result['sid']            = data[test_index]['sid']
result.reset_index(inplace=True)
result.drop(['index'],axis=1,inplace=True)
result_proba = lgb_model.predict_proba(test_x)
a  = pd.DataFrame(pred_proba(result_proba))
result= pd.concat([result,a],axis=1)
result=result.rename(columns={0:'recommend_mode'})
print(len(result))
print(result['recommend_mode'].value_counts())
filename="{:%Y-%m-%d_%H_%M}_sub_beijing.csv".format(datetime.now())
result[['sid', 'recommend_mode']].to_csv(path+'sub/'+filename, index=False)