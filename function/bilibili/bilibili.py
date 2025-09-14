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

#a=pd.read_csv('55.csv')
#b=pd.read_csv('66.csv')
#c=pd.read_csv('33.csv')
#d=pd.read_csv('44.csv')
#df = pd.concat([a, b, c, d], axis=0)
df = pd.read_csv('user_data0914.csv')

group_mapping = {
    '对照组': 'A',
    '实验组1': 'B',
    '实验组2': 'C', 
}

df['new_group'] = df['group'].map(group_mapping)

from sklearn.preprocessing import LabelEncoder

# 创建LabelEncoder对象
le = LabelEncoder()

# 拟合并转换
df['user_type_labelencoded'] = le.fit_transform(df['User_os'])

#df['Favorited_songs_typelabelencoded'] = le.fit_transform(df['Favorited_songs_type'])



X = df.drop(['Userid',
       'is_add'  ,'new_group' ,'group' ,'User_os'
            ], axis=1)
y = df['is_add']
variant = df['new_group'].astype('category').cat.codes 

variant_multi = np.column_stack([(variant == i).astype(float) for i in [0,1,2]])

discrete_features = ['user_type_labelencoded']
continuous_features = ['90days_purchase_time', '90days_per_purchase_price',
       '90days_purchase_amount', '90days_coupon_time', '90days_coupon_ratio',
       'Last_purchase_day']


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

to_pred= cf.effect(X=X_test,T0=0,T1=1)

to_pre= cf.effect(X=X_test,T0=0,T1=2)

print(f"::notice::对照组和实验组1差别为: {to_pred.mean():.8f}")
print(f"::notice::对照组和实验组2差别为: {to_pre.mean():.8f}")

#to_pr= cf.effect(X=X_test,T0=0,T1=3)

#treatment_effects = np.column_stack([to_pred, to_pre, to_pr])  # shape=(n_samples, 3)

