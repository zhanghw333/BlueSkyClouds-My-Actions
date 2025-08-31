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

to_pred= cf.effect(X=X_test,T0=0,T1=1)

to_pre= cf.effect(X=X_test,T0=0,T1=2)
to_pr= cf.effect(X=X_test,T0=0,T1=3)

treatment_effects = np.column_stack([to_pred, to_pre, to_pr])  # shape=(n_samples, 3)

# Step 2: 对每个用户选择最优处理（取最大效应对应的处理编号）
# argmax返回的是列索引（0→T1=1, 1→T1=2, 2→T1=3）
optimal_treatment_idx = np.argmax(treatment_effects, axis=1)  # shape=(n_samples,)

# Step 3: 转换为原始处理编号（从1开始）
recommended_popup = optimal_treatment_idx + 1  # 现在值为1/2/3

# Step 4: 创建包含推荐结果的DataFrame（方便业务集成）
results_df = pd.DataFrame({
    'user_id': range(len(X_test)),  # 替换为真实用户ID
    'features': list(X_test),       # 可选：保存特征用于解释
    'recommended_popup': recommended_popup,
    'effect_value': np.max(treatment_effects, axis=1)  # 最大效应值
})

# 查看前几行示例
continuous_columns = continuous_features

# 2. 离散特征的列名由 OneHotEncoder 生成
#    注意：必须先 fit 才能获取真实的类别信息！
preprocessor.fit(X)  # 如果尚未 fit，需先执行这一步
discrete_columns = preprocessor.named_transformers_['onehotencoder'].get_feature_names_out(input_features=discrete_features)

# 3. 合并所有列名（顺序与 transform 时一致）
all_columns = np.concatenate([continuous_columns, discrete_columns])

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 特征矩阵 + 推荐的处理作为标签
X = X_test  # 特征矩阵 (n_samples × n_features)
y = recommended_popup  # 目标标签 (n_samples,)
columns_list = all_columns.tolist()

#feature_importance = np.abs(np.random.rand(X.shape[1]))  # 模拟特征重要性
#top_features = np.argsort(feature_importance)[::-1][:3]

#x_important = X[:, top_features]
#x_train, x_test, y_train, y_test = train_test_split(x_important, y
, test_size=0.2)

# 训练决策树（控制深度防止过拟合）
dt = DecisionTreeClassifier(
    max_depth=3,          # 根据业务需求调整深度
    min_samples_leaf=100, # 叶节点最小样本数
    random_state=42
)
dt.fit(X, y)
#dt.fit(x_train, y_train)
plt.figure(figsize=(50,40))
plot_tree(
    dt,
    feature_names=all_columns.tolist(),  # 替换为你的特征名
    class_names=['A','B','C'],         # 处理编号
    filled=True,                    # 填充颜色表示纯度
    rounded=True,                   # 圆角矩形
    fontsize=12                    # 字体大小
)

plt.savefig("icon/decision_tree_rules.png")  # 保存图片
plt.show()
