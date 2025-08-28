#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 19:23:09 2025

@author: zhanghanwen
"""

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




#预测处理效应：

