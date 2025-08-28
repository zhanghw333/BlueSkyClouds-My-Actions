#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 17:00:46 2025

@author: zhanghanwen
"""

import pandas  as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

a=pd.read_csv('55.csv')
b=pd.read_csv('66.csv')
c=pd.read_csv('33.csv')
d=pd.read_csv('44.csv')
df = pd.concat([a, b, c, d], axis=0)

group_mapping = {
    '对照组': 'A',
    '实验组1': 'B',
    '实验组2': 'C', 
    '实验组3': 'D'
}

df['new_group'] = df['group'].map(group_mapping)

from sklearn.preprocessing import LabelEncoder

# 创建LabelEncoder对象
le = LabelEncoder()

# 拟合并转换
df['user_type_labelencoded'] = le.fit_transform(df['User_os'])

df['Favorited_songs_typelabelencoded'] = le.fit_transform(df['Favorited_songs_type'])


X = df.drop(['User_id','Is_purchased_member','new_group','group','User_os','Favorited_songs_type'
        ,'Is_exclusive_music', 'Is_vip_music_quality', 'Is_download',
       'Is_confirm'    
            ], axis=1)
y = df['Is_purchased_member']
variant = df['new_group'].astype('category').cat.codes 

variant_multi = np.column_stack([(variant == i).astype(float) for i in [1, 2, 3]])

discrete_features = ['user_type_labelencoded', 'Favorited_songs_typelabelencoded', 'Is_mobile_data',
  'Is_buy_album']
continuous_features = ['Register_days', 'Listening_days', 'Listening_duration'
    ,'Songs_count','Favorited_songs_count','Mobile_data_listening_duration',
    'Search_count']


preprocessor = make_column_transformer(
    (StandardScaler(), continuous_features),
    (OneHotEncoder(handle_unknown='ignore'), discrete_features)
)

# 应用预处理
X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test, variant_train, variant_test = train_test_split(
    X_processed, y, variant, test_size=0.2, random_state=42)


from econml.dml import CausalForestDML

cf = CausalForestDML(

   model_y=RandomForestClassifier(n_estimators=100, random_state=42),
   model_t=RandomForestClassifier(n_estimators=100, random_state=42),
   discrete_treatment=True,
    discrete_outcome=True,
    random_state=42,
    n_estimators=100,  
    max_depth=5,       # 单棵树最大深度
    min_samples_leaf=10,  # 叶节点最小样本数
    min_samples_split=20
)

cf.fit(X=X_train, T=variant_train, Y=y_train)

to_pred=cf.predict(X_processed)
print(f"::notice::模型准确率为: {to_pred}") 





