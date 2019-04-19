#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/19 12:05
# @Author  : zyh
# @Site    : 
# @File    : compute_distance.py
# @Software: PyCharm


import numpy as np
from numpy import linalg as lg
from sklearn.decomposition import PCA
import read_data as reader
import pandas as pd


# 欧氏距离
def euclidean_distance(x, y):
    return 1 / (1.0 + lg.norm(x - y))


# 余弦相似度
def cos_distance(x, y):
    return 0.5 + 0.5 * (np.matmul(x, y) / (lg.norm(x) * lg.norm(y)))


# 皮尔逊系数
def pears_distance(x, y):
    return 0.5 + 0.5 * np.corrcoef(x, y, rowvar=0)[0][1]


# 计算电影标签的相似度矩阵
def similarity_label(labels, method=cos_distance):
    rows = np.shape(labels)[0]
    similarity = np.zeros([rows, rows])
    for i in range(0, rows):
        for j in range(i+1, rows):
            similarity[i][j] = method(labels[i, :], labels[j, :])

    return similarity


# 计算评分相似度矩阵
def similarity_rates(rates, method=cos_distance):
    rates = rates.T
    #pca = PCA(n_components=10)
    #reduce_rates = pca.fit_transform(rates)
    rows = np.shape(rates)[0]
    similarity = np.zeros([rows, rows])
    for i in range(0, rows):
        for j in range(i+1, rows):
            similarity[i][j] = method(rates[i, :], rates[j, :])

    return similarity


# 综合相似度
def similarity_total(label_sim, rate_sim, alpha):
    return alpha * rate_sim + (1 - alpha) * label_sim


if __name__ == '__main__':
    # 标签相似度
    labels = reader.get_labels('./ml-100k/u.item')
    label_sim = similarity_label(labels)

    # 评分相似度
    rates = reader.get_rates_csv('./ml-100k/rate1.csv')
    rate_sim = similarity_rates(rates)

    # 综合相似度
    total_sim = similarity_total(label_sim, rate_sim, 0.7)
    df = pd.DataFrame(total_sim)
    df.to_csv('./ml-100k/total_sim1.csv', header=None, index=None)

