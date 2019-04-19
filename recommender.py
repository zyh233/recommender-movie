#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/19 13:10
# @Author  : zyh
# @Site    : 
# @File    : recommender.py
# @Software: PyCharm

import read_data as reader
import compute_distance as similarity
import pandas as pd
from texttable import Texttable
import numpy as np
from numpy import *


# 推荐引擎， 向user推荐
def recommend(rates, user, total_sim, N=10):
    zero = np.where(rates[user, :] == 0)[0]
    nonzero = np.where(rates[user, :] != 0)[0]
    score = []
    for i in zero:
        sum = 0
        weight = 0
        for j in nonzero:
            if i < j:
                sum += total_sim[i][j] * rates[user][j]
                weight += total_sim[i][j]
            else:
                sum += total_sim[j][i] * rates[user][j]
                weight += total_sim[j][i]
        score.append(sum / weight)
    score = np.array(score)
    args = argsort(-score)
    score = score[args]
    zero = zero[args]
    return score, zero


def get_rates_sims(sim_est=similarity.cos_distance):
    #labels = reader.get_labels('./ml-100k/u.item')
    #label_sim = similarity.similarity_label(labels, method=sim_est)

    # 评分相似度
    rates = reader.get_rates_csv('./ml-100k/rate1.csv')
    #rate_sim = similarity.similarity_rates(rates, method=sim_est)

    # 综合相似度
    total_sim = pd.read_csv('./ml-100k/total_sim1.csv', header=None).values

    return rates, total_sim


if __name__ == '__main__':
    rate, sim = get_rates_sims()
    user = 0
    pre_rates, movies = recommend(rate, user, sim)
    movie_list = reader.getMovieList('./ml-100k/u.item')
    rows = []
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['t', 't', 't'])
    table.set_cols_align(["l", "l", "l"])
    rows.append([u"movie id", u"movie name", u"predict rating"])
    for i in range(0, 20):
        rows.append([movies[i] + 1, movie_list[movies[i] + 1], pre_rates[i]])
    table.add_rows(rows)
    print('user %d predict:' % (user + 1))
    print(table.draw())
