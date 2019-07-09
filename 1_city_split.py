# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:17:18 2019

@author: zhangli
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:21:55 2019

@author: zhangli
"""
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
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb
import os

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import json 
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from itertools import product
import ast
from sklearn.cluster import KMeans

path = '../data_set_phase2/data/'

train_quries1= pd.read_csv(path+ 'train_queries_phase1.csv', parse_dates=['req_time'])
train_quries2= pd.read_csv(path+ 'train_queries_phase2.csv', parse_dates=['req_time'])

train_plans1  = pd.read_csv(path + 'train_plans_phase1.csv', parse_dates=['plan_time'])
train_plans2  = pd.read_csv(path + 'train_plans_phase2.csv', parse_dates=['plan_time'])

train_clicks1  = pd.read_csv(path + 'train_clicks_phase1.csv')
train_clicks2  = pd.read_csv(path + 'train_clicks_phase2.csv')

profiles      = pd.read_csv(path + 'profiles.csv')

test_queries  = pd.read_csv(path + 'test_queries.csv', parse_dates=['req_time'])
test_plans    = pd.read_csv(path + 'test_plans.csv', parse_dates=['plan_time'])


print(train_plans1.shape,train_plans2.shape)

print(train_quries1.shape,train_quries2.shape)

train_quries= pd.concat([train_quries1, train_quries2],axis=0)
train_plans= pd.concat([train_plans1,train_plans2],axis=0)
train_clicks= pd.concat([train_clicks1,train_clicks2],axis=0)

train = train_quries.merge(train_plans, 'left', ['sid'])
test  = test_queries.merge(test_plans, 'left', ['sid'])
train = train.merge(train_clicks, 'left', ['sid'])
train['click_mode'] = train['click_mode'].fillna(0).astype(int)

data  = pd.concat([train, test], ignore_index=True)
data  = data.merge(profiles, 'left', ['pid']) 


def split_od(data):
    data['o_lng'] = data['o'].apply(lambda x: float(x.split(',')[0])).astype(np.float16)
    data['o_lat'] = data['o'].apply(lambda x: float(x.split(',')[1])).astype(np.float16)
    data['d_lng'] = data['d'].apply(lambda x: float(x.split(',')[0])).astype(np.float16)
    data['d_lat'] = data['d'].apply(lambda x: float(x.split(',')[1])).astype(np.float16)
    return data

def city_flag(row):
    if row[1]>35:       # 北京
        return 1
    elif row[1]>27.5:   # 上海
        return 2
    elif row[1]<22.87 and row[0]>113.72:    # 深圳
        return 3
    else:    # 广州
        return 4


data = split_od(data)
data['city_flag_o'] = data[['o_lng','o_lat']].apply(city_flag, axis=1)
data['city_flag_d'] = data[['d_lng','d_lat']].apply(city_flag, axis=1)

data = data[data['city_flag_o'] == data['city_flag_d']]
data = data.drop(['city_flag_d'],axis=1)

beijing= data[data['city_flag_o']==1]
shanghai = data[data['city_flag_o']==2]
shen_guang = data[data['city_flag_o']==3 | data['city_flag_o']==4]


beijing.to_csv('../data_set_phase2/beijing/beijing.csv',index=False)
shanghai.to_csv('../data_set_phase2/shanghai/shanghai.csv',index=False)
shen_guang.to_csv('../data_set_phase2/shen_guang/shen_guang.csv',index=False)
