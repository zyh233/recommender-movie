#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/19 9:30
# @Author  : zyh
# @Site    : 
# @File    : read_data.py
# @Software: PyCharm

import math
import os
import numpy as np
import pandas as pd


# 读取文件，读取以行为单位，每一行是列表里的一个元素
def read_file(filename):
	contents = []
	f = open(filename, "rb")
	contents = f.read().splitlines()
	f.close()
	return contents

# 获取电影标签
def get_labels(filename):
	movies = read_file(filename)
	labels = []
	for movie in movies:
		splits = str(movie, encoding='utf-8').split('|')
		movie_label = [int(splits[5]), int(splits[6]), int(splits[7]), int(splits[8]), int(splits[9]), int(splits[10]), int(splits[11]), int(splits[12]), int(splits[13]), int(splits[14]), int(splits[15]), int(splits[16]),int(splits[17]), int(splits[18]),int(splits[19]), int(splits[20]), int(splits[21]), int(splits[22]), int(splits[23])]
		labels.append(movie_label)

	return np.array(labels)


# 获取评分矩阵
def get_rates(filename):
	data = read_file(filename)
	rates = pd.DataFrame(index=range(0, 943), columns=range(0, 1682), dtype=int, data=0)
	for rate in data:
		splits = str(rate, encoding='utf-8').split('\t')
		print(splits)
		rates[int(splits[1]) - 1][int(splits[0]) - 1] = int(splits[2])

	print(rates.head())
	# 保存评分矩阵
	rates.to_csv('./ml-100k/rate1.csv', header=None, index=None)
	return rates.values


def get_rates_csv(filename):
	df = pd.read_csv(filename, header=None)
	#df.drop(columns="", axis=1)
	return df.values


# 获取电影列表，推荐时使用
def getMovieList(filename):
	contents = read_file(filename)
	movies_info = {}
	for movie in contents:
		single_info = str(movie, encoding='utf-8').split("|")
		movies_info[int(single_info[0])] = single_info[1:2]
	return movies_info


if __name__ == '__main__':
	targts = get_labels('./ml-100k/u.item')
	#np.delete(targts, [0], axis=1)
	print(np.shape(targts))
