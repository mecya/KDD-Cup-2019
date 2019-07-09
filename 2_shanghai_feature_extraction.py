# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:19:49 2019

@author: zhangli
"""


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
import datetime
from sklearn.cluster import KMeans
import math

path = '../data_set_phase2/shanghai/'  #or beijing or shenguang
data= pd.read_csv(path+'shanghai.csv',parse_dates=['req_time','plan_time'])
profiles      = pd.read_csv('../data_set_phase2/data/profiles.csv')

#曼哈顿和欧式距离
data['od_manhattan_distance'] = abs(data['o_lng']-data['d_lng'])+abs(data['o_lat']-data['d_lat'])
data['euclidean']= ((abs(data['o_lng']-data['d_lng']))**2+(abs(data['o_lat']-data['d_lat']))**2)**(1/2)

#另一个距离特征
def GetDistance(lng1, lat1, lng2, lat2):
    EARTH_RADIUS = 6378.137

    lng1 = lng1 * math.pi / 180.0
    lng2 = lng2 * math.pi / 180.0
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0

    dis1 = lat1 - lat2
    dis2 = lng1 - lng2

    s = 2 * math.asin(
        ((math.sin(dis1 / 2)) ** 2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dis2 / 2)) ** 2) ** 0.5)
    s = s * EARTH_RADIUS * 1000
    return s
data['od_manhattan_distance_detail']=data.apply(lambda row:GetDistance(row['o_lng'],row['o_lat'],row['d_lng'],row['d_lat']),axis=1)

def calculate_direction(d_lon,d_lat):
    result= np.zeros(len(d_lon))
    l= np.sqrt(d_lon**2+d_lat**2)
    result[d_lon>0]= (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
    idx = (d_lon<0) & (d_lat>0)
    result[idx]= 180-(180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    idx= (d_lon<0) & (d_lat<0)
    result[idx]= -180-(180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    return result

def add_travel_vector_features(df):    
    df['delta_longitude'] = df.o_lng - df.d_lng
    df['delta_latitude'] = df.o_lat - df.d_lat   
    df['pickup_x'] = np.cos(df.o_lat) * np.cos(df.o_lng)
    df['pickup_y'] = np.cos(df.o_lat) * np.sin(df.o_lng)
    df['pickup_z'] = np.sin(df.o_lat)   
    df['dropoff_x'] = np.cos(df.d_lat) * np.cos(df.d_lng)
    df['dropoff_y'] = np.cos(df.d_lat) * np.sin(df.d_lng)
    df['dropoff_z'] = np.sin(df.d_lat)

add_travel_vector_features(data)    
data['direction'] = calculate_direction(data.delta_longitude, data.delta_latitude)

#经纬度与平均值之间的特征
o_co = data[['o']]
d_co = data[['d']]

o_co.columns = ['co']
d_co.columns = ['co']

all_co = pd.concat([d_co, o_co]).drop_duplicates()
all_co['lng'] = all_co['co'].apply(lambda x: float(x.split(',')[0]))
all_co['lat'] = all_co['co'].apply(lambda x: float(x.split(',')[1]))
lng_mean = all_co['lng'].mean()
lat_mean = all_co['lat'].mean()

lng_mode = all_co['lng'].mode()[0]
lat_mode = all_co['lat'].mode()[0]
data['o_main_centroid_mean_dis'] = abs(
    data['o_lng']-lng_mean)+abs(data['o_lat']-lat_mean)
data['d_main_centroid_mean_dis'] = abs(
    data['d_lng']-lng_mean)+abs(data['d_lat']-lat_mean)

data['o_main_centroid_mode_dis'] = abs(
    data['o_lng']-lng_mode)+abs(data['o_lat']-lat_mode)
data['d_main_centroid_mode_dis'] = abs(
    data['d_lng']-lng_mode)+abs(data['d_lat']-lat_mode)



subway= pd.read_csv('../data_set_phase2/external_data/china_subway.csv', header=None)
shanghai= subway[(subway[1]>120.51) & (subway[1]< 122.12) & (subway[2]>30.40) & (subway[2] < 31.53)]
#beijing= data[(data[1]>115.25) & (data[1]< 117.30) & (data[2]>39.26) & (data[2] < 41.03)]
#shenguang= data[(data[1]>112.57) & (data[1]< 114.3) & (data[2]>22.26) & (data[2] < 23.56)]
print('地铁特征')
#地铁关联特征（构建与最近的地铁站之间的经纬度距离）
#读取地铁数据
shanghai['metro_station_lng'] = shanghai[1]
shanghai['metro_station_lat'] = shanghai[2]
#始终点到地铁的最近距离        
data['o_nearest_subway_dis', 'd_nearest_subway_dis', 'odis2subway','ddis2subway'] = np.nan
for co in tqdm(all_co['co']):
    #始终点到地铁的最近曼哈顿距离 
    lg, la = co.split(',')
    min_dis = (abs(shanghai['metro_station_lng']-float(lg)) + abs(shanghai['metro_station_lat']-float(la))).min()
    data.loc[(data['o'] == co), 'o_nearest_subway_dis'] = min_dis
    data.loc[(data['d'] == co), 'd_nearest_subway_dis'] = min_dis
    min_distance = shanghai.apply(lambda row:GetDistance(row['metro_station_lng'],row['metro_station_lat'],float(lg),float(la)),axis=1).min()
    data.loc[(data['o'] == co), 'odis2subway'] = min_distance
    data.loc[(data['d'] == co), 'ddis2subway'] = min_distance

data['o_d_dis2subway']=data['o_nearest_subway_dis']+data['d_nearest_subway_dis']  
data['o_d_dis2subway1']=data['odis2subway']+data['ddis2subway']             
subway_feature = ['o_d_dis2subway','o_nearest_subway_dis', 'd_nearest_subway_dis', 'o_d_dis2subway1','odis2subway','ddis2subway']  
print('地铁特征数量：', len(subway_feature), subway_feature) 


#原始时间特征上的处理
time_feature = []
for i in ['req_time']:
    data[i + '_hour'] = data[i].dt.hour
    data[i + '_weekday'] = data[i].dt.weekday
    data[i + '_minute'] = data[i].dt.minute
    data[i + '_date_d'] = data['req_time'].dt.strftime('%d').astype(int)
    
    time_feature.append(i + '_hour')
    time_feature.append(i + '_weekday')
    time_feature.append(i + '_minute')   
    time_feature.append(i + '_date_d')  

data['time_diff'] = (data['plan_time']- data['req_time']).dt.seconds

time_feature.append('time_diff')


print('------  构建针对profile的主成分特征。   ------')
x = profiles.drop(['pid'], axis=1).values
svd = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
svd_x = svd.fit_transform(x)
svd_feas = pd.DataFrame(svd_x)
svd_feas.columns = ['svd_fea_{}'.format(i) for i in range(10)]
svd_feas['pid'] = profiles['pid'].values
data = data.merge(svd_feas, on='pid', how='left')

data['plans_json'] = data['plans'].fillna('[]').apply(lambda x: json.loads(x))
def gen_plan_feas(data):
    n                                           = data.shape[0]
    mode_list_feas                              = np.zeros((n, 12))
    max_dist, min_dist, mean_dist, std_dist     = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_price, min_price, mean_price, std_price = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_eta, min_eta, mean_eta, std_eta         = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    min_dist_mode, max_dist_mode, min_price_mode, max_price_mode, min_eta_mode, max_eta_mode, first_mode = \
    np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
  
    mode_texts = []
    for i, plan in tqdm(enumerate(data['plans_json'].values)):
        if len(plan) == 0:
            cur_plan_list   = []
        else:
            cur_plan_list   = plan
        if len(cur_plan_list) == 0:
            mode_list_feas[i, 0] =  1
            first_mode[i]        =  0
            max_dist[i]          = -1
            min_dist[i]          = -1
            mean_dist[i]         = -1
            std_dist[i]          = -1
            max_price[i]         = -1
            min_price[i]         = -1
            mean_price[i]        = -1
            std_price[i]         = -1
            max_eta[i]           = -1
            min_eta[i]           = -1
            mean_eta[i]          = -1
            std_eta[i]           = -1
            min_dist_mode[i]     = -1
            max_dist_mode[i]     = -1
            min_price_mode[i]    = -1
            max_price_mode[i]    = -1
            min_eta_mode[i]      = -1
            max_eta_mode[i]      = -1
            mode_texts.append('word_null')
        else:
            distance_list = []
            price_list = []
            eta_list = []
            mode_list = []
            for tmp_dit in cur_plan_list:
                distance_list.append(int(tmp_dit['distance']))
                if tmp_dit['price'] == '':
                    price_list.append(0)
                else:
                    price_list.append(int(tmp_dit['price']))
                eta_list.append(int(tmp_dit['eta']))
                mode_list.append(int(tmp_dit['transport_mode']))
            mode_texts.append(
                ' '.join(['word_{}'.format(mode) for mode in mode_list]))
            distance_list                = np.array(distance_list)
            price_list                   = np.array(price_list)
            eta_list                     = np.array(eta_list)
            mode_list                    = np.array(mode_list, dtype='int')
            mode_list_feas[i, mode_list] = 1
            distance_sort_idx            = np.argsort(distance_list)
            price_sort_idx               = np.argsort(price_list)
            eta_sort_idx                 = np.argsort(eta_list)
            max_dist[i]                  = distance_list[distance_sort_idx[-1]]
            min_dist[i]                  = distance_list[distance_sort_idx[0]]
            mean_dist[i]                 = np.mean(distance_list)
            std_dist[i]                  = np.std(distance_list)
            max_price[i]                 = price_list[price_sort_idx[-1]]
            min_price[i]                 = price_list[price_sort_idx[0]]
            mean_price[i]                = np.mean(price_list)
            std_price[i]                 = np.std(price_list)
            max_eta[i]                   = eta_list[eta_sort_idx[-1]]
            min_eta[i]                   = eta_list[eta_sort_idx[0]]
            mean_eta[i]                  = np.mean(eta_list)
            std_eta[i]                   = np.std(eta_list)
            first_mode[i]                = mode_list[0]
            max_dist_mode[i]             = mode_list[distance_sort_idx[-1]]
            min_dist_mode[i]             = mode_list[distance_sort_idx[0]]
            max_price_mode[i]            = mode_list[price_sort_idx[-1]]
            min_price_mode[i]            = mode_list[price_sort_idx[0]]
            max_eta_mode[i]              = mode_list[eta_sort_idx[-1]]
            min_eta_mode[i]              = mode_list[eta_sort_idx[0]]
    feature_data                   =  pd.DataFrame(mode_list_feas)
    feature_data.columns           =  ['mode_feas_{}'.format(i) for i in range(12)]
    feature_data['max_dist']       =  max_dist
    feature_data['min_dist']       =  min_dist
    feature_data['mean_dist']      =  mean_dist
    feature_data['std_dist']       =  std_dist
    feature_data['max_price']      = max_price
    feature_data['min_price']      = min_price
    feature_data['mean_price']     = mean_price
    feature_data['std_price']      = std_price
    feature_data['max_eta']        = max_eta
    feature_data['min_eta']        = min_eta
    feature_data['mean_eta']       = mean_eta
    feature_data['std_eta']        = std_eta
    feature_data['max_dist_mode']  = max_dist_mode
    feature_data['min_dist_mode']  = min_dist_mode
    feature_data['max_price_mode'] = max_price_mode
    feature_data['min_price_mode'] = min_price_mode
    feature_data['max_eta_mode']   = max_eta_mode
    feature_data['min_eta_mode']   = min_eta_mode
    feature_data['first_mode']     = first_mode
    print('mode tfidf...')
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(mode_texts)
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    mode_svd = svd_enc.fit_transform(tfidf_vec)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['svd_mode_{}'.format(i) for i in range(10)]
    plan_fea = pd.concat([feature_data, mode_svd], axis=1)
    plan_fea['sid'] = data['sid'].values
    return plan_fea

data_plans = gen_plan_feas(data)
plan_features = [col for col in data_plans.columns if col not in ['sid']]
data = data.merge(data_plans, on='sid', how='left')


data['plans_json'] = data['plans'].fillna('[]').apply(lambda x: json.loads(x))
def gen_plan_pingpu_feas(data):  
    #创建用于放置   mode平均距离、平均价格、平均eta、出现位置排名  共11个模式的   11*4个特征     （为0的模式就不做标识了）    添加model所属的工具模态
    n                                           = data.shape[0]
    #mode_list_feas                              = np.zeros((n, 12))
    model_1_dist,model_1_price,model_1_eta,model_1_rank  =   np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    model_2_dist,model_2_price,model_2_eta,model_2_rank  =   np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    model_3_dist,model_3_price,model_3_eta,model_3_rank  =   np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    model_4_dist,model_4_price,model_4_eta,model_4_rank  =   np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    model_5_dist,model_5_price,model_5_eta,model_5_rank  =   np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    model_6_dist,model_6_price,model_6_eta,model_6_rank  =   np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    model_7_dist,model_7_price,model_7_eta,model_7_rank  =   np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    model_8_dist,model_8_price,model_8_eta,model_8_rank  =   np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    model_9_dist,model_9_price,model_9_eta,model_9_rank  =   np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    model_10_dist,model_10_price,model_10_eta,model_10_rank  =   np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    model_11_dist,model_11_price,model_11_eta,model_11_rank  =   np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    n=0
    for i, plan in tqdm(enumerate(data['plans_json'].values)):
        if len(plan) == 0:
            cur_plan_list   = []
        else:
            cur_plan_list   = plan
        if len(cur_plan_list) == 0:
            model_1_dist[i]=-1
            model_1_price[i]=-1
            model_1_eta[i]=-1
            model_1_rank[i]=-1
            model_2_dist[i]=-1
            model_2_price[i]=-1
            model_2_eta[i]=-1
            model_2_rank[i]=-1
            model_3_dist[i]=-1
            model_3_price[i]=-1
            model_3_eta[i]=-1
            model_3_rank[i]=-1
            model_4_dist[i]=-1
            model_4_price[i]=-1
            model_4_eta[i]=-1
            model_4_rank[i]=-1
            model_5_dist[i]=-1
            model_5_price[i]=-1
            model_5_eta[i]=-1
            model_5_rank[i]=-1
            model_6_dist[i]=-1
            model_6_price[i]=-1
            model_6_eta[i]=-1
            model_6_rank[i]=-1
            model_7_dist[i]=-1
            model_7_price[i]=-1
            model_7_eta[i]=-1
            model_7_rank[i]=-1
            model_8_dist[i]=-1
            model_8_price[i]=-1
            model_8_eta[i]=-1
            model_8_rank[i]=-1
            model_9_dist[i]=-1
            model_9_price[i]=-1
            model_9_eta[i]=-1
            model_9_rank[i]=-1
            model_10_dist[i]=-1
            model_10_price[i]=-1
            model_10_eta[i]=-1
            model_10_rank[i]=-1
            model_11_dist[i]=-1
            model_11_price[i]=-1
            model_11_eta[i]=-1
            model_11_rank[i]=-1
        else:
            
            model_1_dist_list=[]
            model_1_price_list=[]
            model_1_eta_list=[]
            model_2_dist_list=[]
            model_2_price_list=[]
            model_2_eta_list=[]
            model_3_dist_list=[]
            model_3_price_list=[]
            model_3_eta_list=[]
            model_4_dist_list=[]
            model_4_price_list=[]
            model_4_eta_list=[]
            model_5_dist_list=[]
            model_5_price_list=[]
            model_5_eta_list=[]
            model_6_dist_list=[]
            model_6_price_list=[]
            model_6_eta_list=[]
            model_7_dist_list=[]
            model_7_price_list=[]
            model_7_eta_list=[]
            model_8_dist_list=[]
            model_8_price_list=[]
            model_8_eta_list=[]
            model_9_dist_list=[]
            model_9_price_list=[]
            model_9_eta_list=[]
            model_10_dist_list=[]
            model_10_price_list=[]
            model_10_eta_list=[]
            model_11_dist_list=[]
            model_11_price_list=[]
            model_11_eta_list=[]

            mode_list = []
            for tmp_dit in cur_plan_list:
                if tmp_dit['price'] == '':
                    tmp_dit['price']=0
                if tmp_dit['transport_mode']==1:
                    model_1_dist_list.append(int(tmp_dit['distance']))
                    model_1_price_list.append(int(tmp_dit['price']))
                    model_1_eta_list.append(int(tmp_dit['eta']))
                elif tmp_dit['transport_mode']==2:
                    model_2_dist_list.append(int(tmp_dit['distance']))
                    model_2_price_list.append(int(tmp_dit['price']))
                    model_2_eta_list.append(int(tmp_dit['eta']))
                elif tmp_dit['transport_mode']==3:
                    model_3_dist_list.append(int(tmp_dit['distance']))
                    model_3_price_list.append(int(tmp_dit['price']))
                    model_3_eta_list.append(int(tmp_dit['eta']))
                elif tmp_dit['transport_mode']==4:
                    model_4_dist_list.append(int(tmp_dit['distance']))
                    model_4_price_list.append(int(tmp_dit['price']))
                    model_4_eta_list.append(int(tmp_dit['eta']))
                elif tmp_dit['transport_mode']==5:
                    model_5_dist_list.append(int(tmp_dit['distance']))
                    model_5_price_list.append(int(tmp_dit['price']))
                    model_5_eta_list.append(int(tmp_dit['eta']))
                elif tmp_dit['transport_mode']==6:
                    model_6_dist_list.append(int(tmp_dit['distance']))
                    model_6_price_list.append(int(tmp_dit['price']))
                    model_6_eta_list.append(int(tmp_dit['eta']))
                elif tmp_dit['transport_mode']==7:
                    model_7_dist_list.append(int(tmp_dit['distance']))
                    model_7_price_list.append(int(tmp_dit['price']))
                    model_7_eta_list.append(int(tmp_dit['eta']))
                elif tmp_dit['transport_mode']==8:
                    model_8_dist_list.append(int(tmp_dit['distance']))
                    model_8_price_list.append(int(tmp_dit['price']))
                    model_8_eta_list.append(int(tmp_dit['eta']))
                elif tmp_dit['transport_mode']==9:
                    model_9_dist_list.append(int(tmp_dit['distance']))
                    model_9_price_list.append(int(tmp_dit['price']))
                    model_9_eta_list.append(int(tmp_dit['eta']))
                elif tmp_dit['transport_mode']==10:
                    model_10_dist_list.append(int(tmp_dit['distance']))
                    model_10_price_list.append(int(tmp_dit['price']))
                    model_10_eta_list.append(int(tmp_dit['eta']))
                elif tmp_dit['transport_mode']==11:
                    model_11_dist_list.append(int(tmp_dit['distance']))
                    model_11_price_list.append(int(tmp_dit['price']))
                    model_11_eta_list.append(int(tmp_dit['eta']))
                                             
                mode_list.append(int(tmp_dit['transport_mode']))
            
            
            
            mode_list.extend([1,2,3,4,5,6,7,8,9,10,11])        
            mode_list_end  =len(mode_list)-11
                                             
            model_1_dist[i]=np.mean(model_1_dist_list)
            model_1_price[i]=np.mean(model_1_price_list)
            model_1_eta[i]=np.mean(model_1_eta_list)
            if mode_list.index(1)<mode_list_end:
                model_1_rank[i]=mode_list.index(1)
            else:
                model_1_rank[i]=-1   
                                             
            model_2_dist[i]=np.mean(model_2_dist_list)                  
            model_2_price[i]=np.mean(model_2_price_list)
            model_2_eta[i]=np.mean(model_2_eta_list)
            if mode_list.index(2)<mode_list_end:
                model_2_rank[i]=mode_list.index(2)
            else:
                model_2_rank[i]=-1   
                                                              
            model_3_dist[i]=np.mean(model_3_dist_list)
            model_3_price[i]=np.mean(model_3_price_list)
            model_3_eta[i]=np.mean(model_3_eta_list)
            if mode_list.index(3)<mode_list_end:
                model_3_rank[i]=mode_list.index(3)
            else:
                model_3_rank[i]=-1   
                                                              
            model_4_dist[i]=np.mean(model_4_dist_list)
            model_4_price[i]=np.mean(model_4_price_list)
            model_4_eta[i]=np.mean(model_4_eta_list)
            if mode_list.index(4)<mode_list_end:
                model_4_rank[i]=mode_list.index(4)
            else:
                model_4_rank[i]=-1   
                                                              
            model_5_dist[i]=np.mean(model_5_dist_list)
            model_5_price[i]=np.mean(model_5_price_list)
            model_5_eta[i]=np.mean(model_5_eta_list)
            if mode_list.index(5)<mode_list_end:
                model_5_rank[i]=mode_list.index(5)
            else:
                model_5_rank[i]=-1   
                                                              
            model_6_dist[i]=np.mean(model_6_dist_list)
            model_6_price[i]=np.mean(model_6_price_list)
            model_6_eta[i]=np.mean(model_6_eta_list)
            if mode_list.index(6)<mode_list_end:
                model_6_rank[i]=mode_list.index(6)
            else:
                model_6_rank[i]=-1   
                                                              
            model_7_dist[i]=np.mean(model_7_dist_list)
            model_7_price[i]=np.mean(model_7_price_list)
            model_7_eta[i]=np.mean(model_7_eta_list)
            if mode_list.index(7)<mode_list_end:
                model_7_rank[i]=mode_list.index(7)
            else:
                model_7_rank[i]=-1   
                                                              
            model_8_dist[i]=np.mean(model_8_dist_list)
            model_8_price[i]=np.mean(model_8_price_list)
            model_8_eta[i]=np.mean(model_8_eta_list)
            if mode_list.index(8)<mode_list_end:
                model_8_rank[i]=mode_list.index(8)
            else:
                model_8_rank[i]=-1   
                                                              
            model_9_dist[i]=np.mean(model_9_dist_list)
            model_9_price[i]=np.mean(model_9_price_list)
            model_9_eta[i]=np.mean(model_9_eta_list)
            if mode_list.index(9)<mode_list_end:
                model_9_rank[i]=mode_list.index(8)
            else:
                model_9_rank[i]=-1   
                                                              
            model_10_dist[i]=np.mean(model_10_dist_list)
            model_10_price[i]=np.mean(model_10_price_list)
            model_10_eta[i]=np.mean(model_10_eta_list)
            if mode_list.index(10)<mode_list_end:
                model_10_rank[i]=mode_list.index(10)
            else:
                model_10_rank[i]=-1   
                                                              
            model_11_dist[i]=np.mean(model_11_dist_list)
            model_11_price[i]=np.mean(model_11_price_list)
            model_11_eta[i]=np.mean(model_11_eta_list)
            if mode_list.index(11)<mode_list_end:
                model_11_rank[i]=mode_list.index(11)
            else:
                model_11_rank[i]=-1   
                                             
                                             
                                             
    data['plan_model_1_dist']    =  model_1_dist
    data['plan_model_1_price']    =  model_1_price                      
    data['plan_model_1_eta']    =  model_1_eta     
    data['plan_model_1_rank']    =  model_1_rank 
                                             
    data['plan_model_2_dist']    =  model_2_dist
    data['plan_model_2_price']    =  model_2_price                      
    data['plan_model_2_eta']    =  model_2_eta     
    data['plan_model_2_rank']    =  model_2_rank                                             

    data['plan_model_3_dist']    =  model_3_dist
    data['plan_model_3_price']    =  model_3_price                      
    data['plan_model_3_eta']    =  model_3_eta     
    data['plan_model_3_rank']    =  model_3_rank

    data['plan_model_4_dist']    =  model_4_dist
    data['plan_model_4_price']    =  model_4_price                      
    data['plan_model_4_eta']    =  model_4_eta     
    data['plan_model_4_rank']    =  model_4_rank
                                        
    data['plan_model_5_dist']    =  model_5_dist
    data['plan_model_5_price']    =  model_5_price                      
    data['plan_model_5_eta']    =  model_5_eta     
    data['plan_model_5_rank']    =  model_5_rank
                                             
    data['plan_model_6_dist']    =  model_6_dist
    data['plan_model_6_price']    =  model_6_price                      
    data['plan_model_6_eta']    =  model_6_eta     
    data['plan_model_6_rank']    =  model_6_rank
                                            
    data['plan_model_7_dist']    =  model_7_dist
    data['plan_model_7_price']    =  model_7_price                      
    data['plan_model_7_eta']    =  model_7_eta     
    data['plan_model_7_rank']    =  model_7_rank
                                             
    data['plan_model_8_dist']    =  model_8_dist
    data['plan_model_8_price']    =  model_8_price                      
    data['plan_model_8_eta']    =  model_8_eta     
    data['plan_model_8_rank']    =  model_8_rank
                                             
    data['plan_model_9_dist']    =  model_9_dist
    data['plan_model_9_price']    =  model_9_price                      
    data['plan_model_9_eta']    =  model_9_eta     
    data['plan_model_9_rank']    =  model_9_rank
                                             
    data['plan_model_10_dist']    =  model_10_dist
    data['plan_model_10_price']    =  model_10_price                      
    data['plan_model_10_eta']    =  model_10_eta     
    data['plan_model_10_rank']    =  model_10_rank   

    data['plan_model_11_dist']    =  model_11_dist
    data['plan_model_11_price']    =  model_11_price                      
    data['plan_model_11_eta']    =  model_11_eta     
    data['plan_model_11_rank']    =  model_11_rank
                                             
    return data
                                             
new_df=gen_plan_pingpu_feas(data)
data=new_df

import time
#计算与当天四个时刻点的距离，  分钟差距可以为负数


def diff_6_clock(time_point):
    six_clock=str(time_point.year)+'-'+str(time_point.month)+'-'+str(time_point.day)+' '+str(6)+':0:00'
    six_clock=pd.to_datetime(six_clock)
    the_diff=(six_clock- time_point).total_seconds()
    return abs(the_diff/60)


def diff_12_clock(time_point):
    t12_clock=str(time_point.year)+'-'+str(time_point.month)+'-'+str(time_point.day)+' '+str(12)+':0:00'
    t12_clock=pd.to_datetime(t12_clock)
    the_diff=(t12_clock- time_point).total_seconds()
    return abs(the_diff/60)

def diff_18_clock(time_point):
    t18_clock=str(time_point.year)+'-'+str(time_point.month)+'-'+str(time_point.day)+' '+str(18)+':0:00'
    t18_clock=pd.to_datetime(t18_clock)
    the_diff=(t18_clock- time_point).total_seconds()
    return abs(the_diff/60)

def diff_24_clock(time_point):
    t24_clock=str(time_point.year)+'-'+str(time_point.month)+'-'+str(time_point.day)+' '+str(23)+':59:00'
    t24_clock=pd.to_datetime(t24_clock)
    the_diff=(t24_clock- time_point).total_seconds()
    return abs(the_diff/60)

data['diff_6_cloc']=data['req_time'].apply(diff_6_clock)
print('----  第一差距提取结束  ----')
data['diff_12_clock']=data['req_time'].apply(diff_12_clock)
print('----  第二差距提取结束  ----')
data['diff_18_clock']=data['req_time'].apply(diff_18_clock)
print('----  第三差距提取结束  ----')
data['diff_24_clock']=data['req_time'].apply(diff_24_clock)

print('----  第四差距提取结束  ----')


data['req_time_day']=data.apply(lambda row:row['req_time'].day,axis=1) #直接扔掉
data['req_time_hour']=data.apply(lambda row:row['req_time'].hour,axis=1)
data['req_time_minute']=data.apply(lambda row:row['req_time'].minute,axis=1)
data['req_time_weekday']=data.apply(lambda row:row['req_time'].weekday(),axis=1)


#对地点出现次数进行排序     统计那些 经常是导航起点的位置
#第一步，首先获取每个o 出现的次数

o_appear_count=list(data.groupby(by=['o']).size())
d_appear_count=list(data.groupby(by=['d']).size())

the_query_o_count_df=pd.DataFrame()
the_query_d_count_df=pd.DataFrame()

o_list=[]
for name,group in data.groupby(by=['o']):
    o_list.append(name)
    
d_list=[]
for name,group in data.groupby(by=['d']):
    d_list.append(name)
    
the_query_o_count_df['o']=o_list
the_query_d_count_df['d']=d_list
the_query_o_count_df['o_appear_count']=o_appear_count
the_query_d_count_df['d_appear_count']=d_appear_count



data=data.merge(the_query_o_count_df, 'left', ['o']) 
data=data.merge(the_query_d_count_df, 'left', ['d']) 

print(data.columns.values)
print('-------    构建坐标计数完成    -------')
data['o_appear_count_rank'] = data['o_appear_count'].rank() / float(data.shape[0])
data['d_appear_count_rank'] = data['d_appear_count'].rank() / float(data.shape[0])
data['o_appear_count_rank_buguiyi'] = data['o_appear_count'].rank() 
data['d_appear_count_rank_buguiyi'] = data['d_appear_count'].rank() 
print('-------    排序完成    -------')


#构建od对出现次数 统计特征，并对该特征进行排序，尝试效果。
data['od_couple']=data.apply(lambda row:(row['o']+'_'+row['d']),axis=1) 

od_appear_count=list(data.groupby(by=['od_couple']).size())

the_query_od_count_df=pd.DataFrame()

od_list=[]
for name,group in data.groupby(by=['od_couple']):
    od_list.append(name)
    
the_query_od_count_df['od_couple']=od_list
the_query_od_count_df['od_couple_count']=od_appear_count

data=data.merge(the_query_od_count_df, 'left', ['od_couple']) 

print(data.columns.values)
print('-------    构建坐标计数完成    -------')
data['od_couple_rank'] = data['od_couple_count'].rank() / float(data.shape[0])
data['od_couple_rank_buguiyi'] = data['od_couple_count'].rank() 
print('-------    排序完成    -------')


#第二波排序特征， 对平展后的每个进行  除了rank外的进行整体的排序飘絮
for i in range(1,12):
    data['plan_model_'+str(i)+'_dist'+'_rank']=data['plan_model_'+str(i)+'_dist'].rank() 
    data['plan_model_'+str(i)+'_eta'+'_rank']=data['plan_model_'+str(i)+'_eta'].rank()    
    data['plan_model_'+str(i)+'_price'+'_rank']=data['plan_model_'+str(i)+'_price'].rank() 
    data['plan_model_'+str(i)+'_rank'+'_rank']=data['plan_model_'+str(i)+'_rank'].rank() 
    
    data['plan_model_'+str(i)+'_dist'+'_rank_guiyi']=data['plan_model_'+str(i)+'_dist'].rank()  / float(data.shape[0])
    data['plan_model_'+str(i)+'_eta'+'_rank_guiyi']=data['plan_model_'+str(i)+'_eta'].rank()  / float(data.shape[0])   
    data['plan_model_'+str(i)+'_price'+'_rank_guiyi']=data['plan_model_'+str(i)+'_price'].rank()  / float(data.shape[0])
    data['plan_model_'+str(i)+'_rank'+'_rank_guiyi']=data['plan_model_'+str(i)+'_rank'].rank()  / float(data.shape[0])
    


#聚类特征
def cluster_features(data):
    o_co = data[['o']]
    d_co = data[['d']]

    o_co.columns = ['co']
    d_co.columns = ['co']


    data['o_cluster'] = np.nan
    data['d_cluster'] = np.nan

    all_co = pd.concat([d_co, o_co])['co'].unique()
    X = pd.DataFrame()
    X['lng'] = pd.Series(all_co).apply(lambda x: float(x.split(',')[0]))
    X['lat'] = pd.Series(all_co).apply(lambda x: float(x.split(',')[0]))
    clf_KMeans = KMeans(n_clusters=11)#构造聚类器
    cluster = clf_KMeans.fit_predict(X)#聚类
    index = 0
    for co in tqdm(all_co):
        data.loc[(data['o'] == co), 'o_cluster'] = cluster[index]
        data.loc[(data['d'] == co), 'd_cluster'] = cluster[index]
        index +=1
    return data

data= cluster_features(data)

#协同特征

# 获取出发地热度
def get_sloc_count(train, result):
    sloc_count = train.groupby('o', as_index=False)['pid'].agg({'sloc_count': 'count'})
    result = pd.merge(result, sloc_count, on='o', how='left')
    return result

# 获取目的地作为出发地的热度
def get_eloc_as_sloc_count(train, result):
    eloc_as_sloc_count = train.groupby('o', as_index=False)['pid'].agg({'eloc_as_sloc_count': 'count'})
    eloc_as_sloc_count.rename(columns={'o': 'd'}, inplace=True)
    result = pd.merge(result, eloc_as_sloc_count, on='d', how='left')
    return result
# 获取出发地作为目的地的热度
def get_sloc_as_eloc_count(train, result):
    train = train[~train.d.isnull()]
    sloc_as_eloc_count = train.groupby('d', as_index=False)['pid'].agg({'sloc_as_eloc_count': 'count'})
    sloc_as_eloc_count.rename(columns={'d': 'o'}, inplace=True)
    result = pd.merge(result, sloc_as_eloc_count, on='o', how='left')
    return result
# 获取目的地的热度
def get_eloc_count(train, result):
    train = train[~train.d.isnull()]
    eloc_count = train.groupby('d', as_index=False)['sid'].agg({'eloc_count': 'count'})
    result = pd.merge(result, eloc_count, on='d', how='left')
    return result

# 获取用户目的地点作为出发地的次数
def get_user_eloc_as_sloc_count(train, result):
    user_eloc_as_sloc_count = train.groupby(['pid', 'o'], as_index=False)['pid'].agg({'user_eloc_as_sloc_count': 'count'})
    user_eloc_as_sloc_count.rename(columns={'o': 'd'}, inplace=True)
    result = pd.merge(result, user_eloc_as_sloc_count, on=['pid', 'd'], how='left')
    return result
# 获取用户出发地点作为目的地的次数
def get_user_sloc_as_eloc_count(train, result):
    train = train[~train.d.isnull()]
    user_sloc_as_eloc_count = train.groupby(['pid', 'd'], as_index=False)['pid'].agg({'user_sloc_as_eloc_count': 'count'})
    user_sloc_as_eloc_count.rename(columns={'d': 'o'}, inplace=True)
    result = pd.merge(result, user_sloc_as_eloc_count, on=['pid', 'o'], how='left')
    return result
def get_user_eloc_count(train, result):
    train = train[~train.d.isnull()]
    user_eloc_count = train.groupby(['pid', 'd'], as_index=False)['pid'].agg({'user_eloc_count': 'count'})
    result = pd.merge(result, user_eloc_count, on=['pid', 'd'], how='left')
    return result
# 获取用户从某个地方出发的历史出行次数
def get_user_sloc_count(train, result):
    user_sloc_count = train.groupby(['pid', 'o'], as_index=False)['pid'].agg({'user_sloc_count': 'count'})
    result = pd.merge(result, user_sloc_count, on=['pid', 'o'], how='left')
    return result


data=get_sloc_count(data[['o','d','sid','pid','req_time_hour','od_manhattan_distance','od_manhattan_distance_detail']], data)
data=get_eloc_as_sloc_count(data[['o','d','sid','pid','req_time_hour','od_manhattan_distance','od_manhattan_distance_detail']], data)
data=get_sloc_as_eloc_count(data[['o','d','sid','pid','req_time_hour','od_manhattan_distance','od_manhattan_distance_detail']], data)
data=get_eloc_count(data[['o','d','sid','pid','req_time_hour','od_manhattan_distance','od_manhattan_distance_detail']], data)
data=get_user_eloc_as_sloc_count(data[['o','d','sid','pid','req_time_hour','od_manhattan_distance','od_manhattan_distance_detail']], data)
data=get_user_sloc_as_eloc_count(data[['o','d','sid','pid','req_time_hour','od_manhattan_distance','od_manhattan_distance_detail']], data)
data=get_user_eloc_count(data[['o','d','sid','pid','req_time_hour','od_manhattan_distance','od_manhattan_distance_detail']], data)
data=get_user_sloc_count(data[['o','d','sid','pid','req_time_hour','od_manhattan_distance','od_manhattan_distance_detail']], data)


# 获取地址对的协同过滤信息
def get_loc_filter(train, result):
    sloc_elocs, eloc_slocs = {}, {}
    for i in tqdm.tqdm(train[['o', 'd']].values):
        if i[0] not in sloc_elocs:
            sloc_elocs[i[0]] = {}
        if i[1] not in sloc_elocs[i[0]]:
            sloc_elocs[i[0]][i[1]] = 0
        sloc_elocs[i[0]][i[1]] += 1
        if i[1] not in eloc_slocs:
            eloc_slocs[i[1]] = {}
        if i[0] not in eloc_slocs[i[1]]:
            eloc_slocs[i[1]][i[0]] = 0;
        eloc_slocs[i[1]][i[0]] += 1
    sloc_list, eloc_list, sloc_eloc_common_eloc_count, sloc_eloc_common_sloc_count, sloc_eloc_common_conn1_count, sloc_eloc_common_conn2_count = [], [], [], [], [], []
    for i in tqdm.tqdm(result[['o', 'd']].drop_duplicates().values):
        sloc_list.append(i[0])
        eloc_list.append(i[1])
        # 获取地址对在历史记录中共有的目的地数目
        common_eloc_count = 0
        if (i[0] in sloc_elocs) and (i[1] in sloc_elocs):
            sloc_eloc_common_eloc_set = sloc_elocs[i[0]].keys() & sloc_elocs[i[1]].keys()
            for common_eloc in sloc_eloc_common_eloc_set:
                common_eloc_count = common_eloc_count + sloc_elocs[i[0]][common_eloc] + sloc_elocs[i[1]][common_eloc]
        sloc_eloc_common_eloc_count.append(common_eloc_count)
        # 获取地址对在历史记录中共有的出发地数目
        common_sloc_count = 0
        if (i[0] in eloc_slocs) and (i[1] in eloc_slocs):
            sloc_eloc_common_sloc_set = eloc_slocs[i[0]].keys() & eloc_slocs[i[1]].keys()
            for common_sloc in sloc_eloc_common_sloc_set:
                common_sloc_count = common_sloc_count + eloc_slocs[i[0]][common_sloc] + eloc_slocs[i[1]][common_sloc]
        sloc_eloc_common_sloc_count.append(common_sloc_count)
        # 获取地址对在历史记录中共有的连接点数目(出发点->xx->目的地)
        common_conn1_count = 0
        if (i[0] in sloc_elocs) and (i[1] in eloc_slocs):
            sloc_eloc_common_conn1_set = sloc_elocs[i[0]].keys() & eloc_slocs[i[1]].keys()
            for common_conn1 in sloc_eloc_common_conn1_set:
                common_conn1_count = common_conn1_count + sloc_elocs[i[0]][common_conn1] + eloc_slocs[i[1]][common_conn1]
        sloc_eloc_common_conn1_count.append(common_conn1_count)
        # 获取地址对在历史记录中共有的连接点数目(出发点<-xx<-目的地)
        common_conn2_count = 0
        if (i[0] in eloc_slocs) and (i[1] in sloc_elocs):
            sloc_eloc_common_conn2_set = eloc_slocs[i[0]].keys() & sloc_elocs[i[1]].keys()
            for common_conn2 in sloc_eloc_common_conn2_set:
                common_conn2_count = common_conn2_count + eloc_slocs[i[0]][common_conn2] + sloc_elocs[i[1]][common_conn2]
        sloc_eloc_common_conn2_count.append(common_conn2_count)
    loc_filter = pd.DataFrame({"o": sloc_list, "d": eloc_list, "sloc_eloc_common_eloc_count": sloc_eloc_common_eloc_count, "sloc_eloc_common_sloc_count": sloc_eloc_common_sloc_count, "sloc_eloc_common_conn1_count": sloc_eloc_common_conn1_count, "sloc_eloc_common_conn2_count": sloc_eloc_common_conn2_count})
    result = pd.merge(result, loc_filter, on=['o', 'd'], how='left')
    result['sloc_eloc_common_eloc_rate'] = result['sloc_eloc_common_eloc_count']/(result['sloc_count']+result['eloc_as_sloc_count'])
    result['sloc_eloc_common_sloc_rate'] = result['sloc_eloc_common_sloc_count']/(result['sloc_as_eloc_count']+result['eloc_count'])
    result['sloc_eloc_common_conn1_rate'] = result['sloc_eloc_common_conn1_count']/(result['sloc_count']+result['eloc_count'])
    result['sloc_eloc_common_conn2_rate'] = result['sloc_eloc_common_conn2_count']/(result['sloc_as_eloc_count']+result['eloc_as_sloc_count'])
    return result


# 获取用户地址对的协同过滤信息
def get_user_loc_filter(train, result):
    user_sloc_elocs, user_eloc_slocs = {}, {}
    for i in tqdm.tqdm(train[['pid', 'o', 'd']].values):
        if i[0] not in user_sloc_elocs:
            user_sloc_elocs[i[0]] = {}
        if i[1] not in user_sloc_elocs[i[0]]:
            user_sloc_elocs[i[0]][i[1]] = {}
        if i[2] not in user_sloc_elocs[i[0]][i[1]]:
            user_sloc_elocs[i[0]][i[1]][i[2]] = 0
        user_sloc_elocs[i[0]][i[1]][i[2]] += 1
        if i[0] not in user_eloc_slocs:
            user_eloc_slocs[i[0]] = {}
        if i[2] not in user_eloc_slocs[i[0]]:
            user_eloc_slocs[i[0]][i[2]] = {};
        if i[1] not in user_eloc_slocs[i[0]][i[2]]:
            user_eloc_slocs[i[0]][i[2]][i[1]] = 0
        user_eloc_slocs[i[0]][i[2]][i[1]] += 1
    user_list, user_sloc_list, user_eloc_list, user_sloc_eloc_common_eloc_count, user_sloc_eloc_common_sloc_count, user_sloc_eloc_common_conn1_count, user_sloc_eloc_common_conn2_count = [], [], [], [], [], [], []
    for i in tqdm.tqdm(result[['pid', 'o', 'd']].drop_duplicates().values):
        user_list.append(i[0])
        user_sloc_list.append(i[1])
        user_eloc_list.append(i[2])
        # 获取地址对在用户历史记录中共有的目的地数目
        user_common_eloc_count = 0
        if (i[0] in user_sloc_elocs) and (i[1] in user_sloc_elocs[i[0]]) and (i[2] in user_sloc_elocs[i[0]]):
            user_sloc_eloc_common_eloc_set = user_sloc_elocs[i[0]][i[1]].keys() & user_sloc_elocs[i[0]][i[2]].keys()
            for user_common_eloc in user_sloc_eloc_common_eloc_set:
                user_common_eloc_count = user_common_eloc_count + user_sloc_elocs[i[0]][i[1]][user_common_eloc] + user_sloc_elocs[i[0]][i[2]][user_common_eloc]
        user_sloc_eloc_common_eloc_count.append(user_common_eloc_count)
        # 获取地址对在用户历史记录中共有的出发地数目
        user_common_sloc_count = 0
        if (i[0] in user_eloc_slocs) and (i[1] in user_eloc_slocs[i[0]]) and (i[2] in user_eloc_slocs[i[0]]):
            user_sloc_eloc_common_sloc_set = user_eloc_slocs[i[0]][i[1]].keys() & user_eloc_slocs[i[0]][i[2]].keys()
            for user_common_sloc in user_sloc_eloc_common_sloc_set:
                user_common_sloc_count = user_common_sloc_count + user_eloc_slocs[i[0]][i[1]][user_common_sloc] + user_eloc_slocs[i[0]][i[2]][user_common_sloc]
        user_sloc_eloc_common_sloc_count.append(user_common_sloc_count)
        # 获取地址对在用户历史记录中共有的连接点数目(出发点->xx->目的地)
        user_common_conn1_count = 0
        if (i[0] in user_sloc_elocs) and (i[1] in user_sloc_elocs[i[0]]) and (i[0] in user_eloc_slocs) and (i[2] in user_eloc_slocs[i[0]]):
            user_sloc_eloc_common_conn1_set = user_sloc_elocs[i[0]][i[1]].keys() & user_eloc_slocs[i[0]][i[2]].keys()
            for user_common_conn1 in user_sloc_eloc_common_conn1_set:
                user_common_conn1_count = user_common_conn1_count + user_sloc_elocs[i[0]][i[1]][user_common_conn1] + user_eloc_slocs[i[0]][i[2]][user_common_conn1]
        user_sloc_eloc_common_conn1_count.append(user_common_conn1_count)
        # 获取地址对在用户历史记录中共有的连接点数目(出发点<-xx<-目的地)
        user_common_conn2_count = 0
        if (i[0] in user_eloc_slocs) and (i[1] in user_eloc_slocs[i[0]]) and (i[0] in user_sloc_elocs) and (i[2] in user_sloc_elocs[i[0]]):
            user_sloc_eloc_common_conn2_set = user_eloc_slocs[i[0]][i[1]].keys() & user_sloc_elocs[i[0]][i[2]].keys()
            for user_common_conn2 in user_sloc_eloc_common_conn2_set:
                user_common_conn2_count = user_common_conn2_count + user_eloc_slocs[i[0]][i[1]][user_common_conn2] + user_sloc_elocs[i[0]][i[2]][user_common_conn2]
        user_sloc_eloc_common_conn2_count.append(user_common_conn2_count)
    user_loc_filter = pd.DataFrame({"pid": user_list, "o": user_sloc_list, "d": user_eloc_list, "user_sloc_eloc_common_eloc_count": user_sloc_eloc_common_eloc_count, "user_sloc_eloc_common_sloc_count": user_sloc_eloc_common_sloc_count, "user_sloc_eloc_common_conn1_count": user_sloc_eloc_common_conn1_count, "user_sloc_eloc_common_conn2_count": user_sloc_eloc_common_conn2_count})
    result = pd.merge(result, user_loc_filter, on=['pid', 'o', 'd'], how='left')
    result['user_sloc_eloc_common_eloc_rate'] = result['user_sloc_eloc_common_eloc_count']/(result['user_sloc_count']+result['user_eloc_as_sloc_count'])
    result['user_sloc_eloc_common_sloc_rate'] = result['user_sloc_eloc_common_sloc_count']/(result['user_sloc_as_eloc_count']+result['user_eloc_count'])
    result['user_sloc_eloc_common_conn1_rate'] = result['user_sloc_eloc_common_conn1_count']/(result['user_sloc_count']+result['user_eloc_count'])
    result['user_sloc_eloc_common_conn2_rate'] = result['user_sloc_eloc_common_conn2_count']/(result['user_sloc_as_eloc_count']+result['user_eloc_as_sloc_count'])
    return result
data=get_loc_filter(data[['pid','o','d']],data)
data=get_user_loc_filter(data[['pid','o','d']],data)
#############################################################################################
#仿照FM构建的cos距离特征

def cal_cos_dis(temp1, temp2):
    dot = np.sum(temp1 * temp2, axis=1)
    norm = np.sqrt(np.sum(temp1 * temp1, axis=1)) * np.sqrt(np.sum(temp2 * temp2, axis=1))
    return dot / norm, dot, norm


def cal_pearson_dist(temp1, temp2):
    e1 = np.mean(temp1, axis=1)
    e2 = np.mean(temp2, axis=1)

    e12 = np.mean(temp1 * temp2, axis=1)
    e11 = np.mean(temp1 * temp1, axis=1)
    e22 = np.mean(temp2 * temp2, axis=1)
    return (e12 - e1 * e2) / (np.sqrt(e11 - e1 * e1) * np.sqrt(e22 - e2 * e2) + 1e-8)


def add_cos_dis(data):
    # 把每个mode的price，eta，dist看成一个三维向量，然后求其模，求两两之间的cos相似度
    dist_feat = ['plan_model_%d_dist' % i for i in range(1, 12)]
    price_feat = ['plan_model_%d_price' % i for i in range(1, 12)]
    eta_feat = ['plan_model_%d_eta' % i for i in range(1, 12)]

    plans = data[dist_feat + price_feat + eta_feat]
    plans = plans.fillna(-1)

    plans[plans==-1] = 99999999
    k = 0
    for i in range(1, 12):
        temp1 = plans[['plan_model_%d_dist' % i, 'plan_model_%d_price' % i, 'plan_model_%d_eta' % i]].values
        data['norm_%d_plan_i' % i] = np.sqrt(np.sum(np.power(temp1, 2), axis=1))

        for j in range(i+1, 12):
            # 向量的基本运算都用上?
            temp2 = plans[['plan_model_%d_dist' % j, 'plan_model_%d_price' % j, 'plan_model_%d_eta' % j]].values
            cos, dot, norm = cal_cos_dis(temp1, temp2)
            data['cos_%d_plan' % k] = cos
            k += 1

    return data

data= add_cos_dis(data)


#类似统计的特征
# 超强特
new_feature = []
for col in ['o','d','od_couple','o_lng','d_lng','o_lat','d_lat','req_time_hour','req_time_weekday','weekend']:
    print(col)

    new_feature.append('%s_count_per_pid'%col)
    temp = data.groupby(['pid',col]).size().rename('%s_count_per_pid'%col)
    data = data.merge(temp, on=['pid',col], how='left')

for col1 in ['o','d','od_couple','o_lng','d_lng','o_lat','d_lat',]:
    for col2 in ['req_time_hour','req_time_weekday','weekend']:
        print(col1, col2)
        temp = data.groupby([col1,col2]).size().rename('%s_count_per_%s'%(col1, col2))
        data = data.merge(temp, on=[col1,col2], how='left')
        new_feature.append('%s_count_per_%s'%(col1, col2))        


######################################   原始变换特征    ######################################
#data=new_df
#这里是自己组建需要使用的特征    baseline特征
plan_features      = ['mode_feas_0', 'mode_feas_1', 'mode_feas_2', 'mode_feas_3', 'mode_feas_4', 'mode_feas_5', 'mode_feas_6', 'mode_feas_7', 'mode_feas_8', 'mode_feas_9', 'mode_feas_10', 'mode_feas_11', 'max_dist', 'min_dist', 'mean_dist', 'std_dist', 'max_price', 'min_price', 'mean_price', 'std_price', 'max_eta', 'min_eta', 'mean_eta', 'std_eta', 'max_dist_mode', 'min_dist_mode', 'max_price_mode', 'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'svd_mode_0', 'svd_mode_1', 'svd_mode_2', 'svd_mode_3', 'svd_mode_4', 'svd_mode_5', 'svd_mode_6', 'svd_mode_7', 'svd_mode_8', 'svd_mode_9']
profile_feature    = ['p' + str(i) for i in range(66)]
origin_num_feature = ['o_lng', 'o_lat', 'd_lng', 'd_lat']+profile_feature
cate_feature       = ['pid']  


#原始编码信息
pingzhan_dist_feature=['plan_model_'+str(i)+'_dist'  for i in range(1,12)]
pingzhan_price_feature=['plan_model_'+str(i)+'_price'  for i in range(1,12)]
pingzhan_rank_feature=['plan_model_'+str(i)+'_rank'  for i in range(1,12)]


base_feature=pingzhan_dist_feature+pingzhan_price_feature+pingzhan_rank_feature
##############################################聚类特征##################################
od_features=['o_cluster','d_cluster']

######################################   日期特征    ######################################
#日期特征添加 
time_diff_feature=['time_diff']
time_clock_feat=['req_time_hour','req_time_minute','req_time_weekday']
time_clock_diff=['diff_6_cloc','diff_12_clock','diff_18_clock','diff_24_clock']
time_jiaocha_detail=[]    #八成降分  去掉了
time_feature = time_diff_feature+time_clock_feat+time_clock_diff +time_jiaocha_detail


######################################   距离特征    ######################################
#距离特征添加         
subway_feature = ['o_d_dis2subway','o_nearest_subway_dis', 'd_nearest_subway_dis', 'o_d_dis2subway1','odis2subway','ddis2subway']  
distance_center_feature=['od_manhattan_distance_detail','o_main_centroid_mean_dis','d_main_centroid_mean_dis','o_main_centroid_mode_dis','d_main_centroid_mode_dis']
distance_feature_1= ['od_manhattan_distance','euclidean','delta_longitude','delta_latitude','pickup_x','pickup_y','pickup_z','dropoff_x','dropoff_y','dropoff_z','direction']

distance_feature = distance_center_feature + subway_feature + distance_feature_1
#distance_feature=subway_feature+distance_center_feature
######################################  协同特征   ###########################################################################

xietong_feature= ['sloc_count','eloc_as_sloc_count', 'sloc_as_eloc_count', 'eloc_count','user_eloc_as_sloc_count','user_sloc_as_eloc_count','user_eloc_count','user_sloc_count','user_sloc_count',
                  'sloc_eloc_common_eloc_count','sloc_eloc_common_sloc_count','sloc_eloc_common_conn1_count','sloc_eloc_common_conn2_count','sloc_eloc_common_eloc_rate','sloc_eloc_common_sloc_rate','sloc_eloc_common_conn1_rate','sloc_eloc_common_conn2_rate',
                  'user_sloc_eloc_common_eloc_count','user_sloc_eloc_common_sloc_count','user_sloc_eloc_common_conn1_count','user_sloc_eloc_common_conn2_count','user_sloc_eloc_common_eloc_rate','user_sloc_eloc_common_sloc_rate','user_sloc_eloc_common_conn1_rate','user_sloc_eloc_common_conn2_rate']

########################################  cos距离特征  ############################################
cos_distance= ['cos_%d_plan' % k for k in range(145)]

######################################   统计特征    ######################################
statistics_feature=[]
######################################   排序特征    ######################################
#排序特征添加
#位置点出行情况排序
od_apper_rank=['o_appear_count', 'd_appear_count', 'o_appear_count_rank',
       'd_appear_count_rank','o_appear_count_rank_buguiyi', 'd_appear_count_rank_buguiyi']
od_couple_rank=['od_couple_count']

#对自己平展方式效果的排序
pingzhan_dist_rank_feature=['plan_model_'+str(i)+'_dist_rank'  for i in range(1,12)]
pingzhan_price_rank_feature=['plan_model_'+str(i)+'_price_rank'  for i in range(1,12)]
pingzhan_eta_rank_feature=['plan_model_'+str(i)+'_eta_rank'  for i in range(1,12)]
pingzhan_rank_rank_feature=['plan_model_'+str(i)+'_rank_rank'  for i in range(1,12)]

plan_pingzhan_static_rank=pingzhan_dist_rank_feature+pingzhan_price_rank_feature+pingzhan_eta_rank_feature+pingzhan_rank_rank_feature


rank_feature=od_apper_rank +od_couple_rank+plan_pingzhan_static_rank
######################################   个人属性特征    ######################################

profile=[]
######################################   特征拼接    ######################################

#特征拼接
feature        =  new_feature + origin_num_feature + plan_features + cate_feature + time_feature + distance_feature + xietong_feature + cos_distance + profile + base_feature + rank_feature + od_features

#删除一部分  低重要度
feature.remove('plan_model_6_price_rank')
feature.remove('plan_model_5_price_rank')
feature.remove('plan_model_5_price')
feature.remove('plan_model_6_price')
feature.remove('plan_model_3_price_rank')
feature.remove('plan_model_3_price')

feature.remove('plan_model_9_rank_rank')
feature.remove('plan_model_6_rank_rank')
feature.remove('plan_model_5_rank_rank')
feature.remove('plan_model_8_rank_rank')
feature.remove('plan_model_10_rank_rank')
feature.remove('plan_model_11_rank_rank')



data[feature].to_csv(path+'shanghai_features.csv',index=False)
#the same to beijing_features, shengunag_features

