# -*- coding: utf-8 -*-
"""
利用感知器（Perceptron）对鸢尾花进行二/多元分类

"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 导入鸢尾花数据集
iris = datasets.load_iris() 

#X=iris.data[:100,:]
#y=iris.target[:100]

# 获得其特征向量
X = iris.data 
# 获得样本标签
y = iris.target 


# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 数据预处理 标准化
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# 感知器分类
classifier=Perceptron(n_iter=40,eta0=0.1,random_state=0)
classifier.fit(X_train_std,y_train)

# 预测
y_pred=classifier.predict(X_test_std)

# 输出分类准确率
print('感知器对鸢尾花数据集分类准确率为:%.4f'%accuracy_score(y_test,y_pred))

