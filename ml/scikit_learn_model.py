#!/usr/bin/env python
#--coding=utf-8

"""
@desc: scikit-learn 主要模块和使用方法
@author: hongxingfan
@date: 2016-09-13 星期二
"""

import numpy as np
import urllib
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def load_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"

    raw_data = urllib.urlopen(url)

    dataset = np.loadtxt(raw_data, delimiter=",")

    x = dataset[:, 0:7]
    y = dataset[:, 8]

    print("\n%s load_data %s" % ("="*15, "="*15))

    print("type x is: %s" % (type(x)))
    print("type y is: %s" % (type(y)))

    print("x shape is: (%s, %s)" % (x.shape[0], x.shape[1]))
    print("y shape is: (%s)" % (y.shape[0]))

    print("x is:\n %s" %(x))
    print("y is:\n %s" %(y))

    return (x, y)

def normalization_data(x):
    #-- http://webdataanalysis.net/data-analysis-method/data-normalization/
    #-- 规则化, 常用x=(x - minX)/(maxX - minX)
    normalization_x = preprocessing.normalize(x)

    #-- 标准化, 让数据满足正态分布, 即均值是0, 方差是1
    standardize_x = preprocessing.scale(x)

    print("\n%s normalization_data %s" % ("="*15, "="*15))

    print("normalization_x is:\n %s" %(normalization_x))
    print("standardize_x is:\n %s" %(standardize_x))

def feature_seletion(x, y):
    model = ExtraTreesClassifier()
    model.fit(x, y)

    print("\n%s feature_seletion %s" % ("="*15, "="*15))
    print(model.feature_importances_)

def logistic_regression(x, y):
    model = LogisticRegression()
    model.fit(x, y)

    print("\n%s logistic_regression %s" % ("="*15, "="*15))
    print("model is:\n %s" % (model))

    predict = model.predict(x)

    print("classification_report is:\n %s" % (metrics.classification_report(y, predict)))

    #-- 混淆矩阵, 判断真是类别和预测类别的数量
    print("confusion_matrix is:\n %s" % (metrics.confusion_matrix(y, predict)))

def naive_bayes(x, y):
    model = GaussianNB()
    model.fit(x, y)
    
    print("\n%s naive_bayes %s" % ("="*15, "="*15))
    print("model is:\n %s" % (model))
    
    predict = model.predict(x)
    print("classification_report is:\n %s" % metrics.classification_report(y, predict))
    print("confusion_matrix is:\n %s" % metrics.confusion_matrix(y, predict))


if __name__ == "__main__":
    x, y = load_data()

    normalization_data(x)

    feature_seletion(x, y)

    logistic_regression(x, y)

    naive_bayes(x, y)
