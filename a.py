# Python file
# Name  :   a
# Create:   2017/11/02 14:07
# Author:   Ma
# Contact   1033516561@qq.com

# Quote :   

# Begin

# mcv             mean corpuscular volume
# alkphos         alkaline phosphotase
# sgpt            alamine aminotransferase
# sgot            aspartate aminotransferase
# gammagt         gamma-glutamyl transpeptidase
# drinks          number of half-pint equivalents of alcoholic beverages drunk per day
# selector        field used to split data into two sets


from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame
from numpy import mean, median, var, std
import statistics
from pylab import plot, show
import pylab as pl
import random
import math

localfn='D:\\workspace\\analyze\\bupa.csv'
dataSet = pandas.read_csv(localfn, names=['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector'])

dataSet1 = dataSet['mcv'].head(100)

# 随机将数据生成两个集合 方式：将该数组随机打乱顺序 再分成两个集合
random.shuffle(dataSet1)
data1 = pandas.DataFrame(dataSet1.head(50))
data2 = pandas.DataFrame(dataSet1.tail(50))

# 修正样本方差
s1_modified = float(data1.std())
s2_modified = float(data2.std())
s1_mean = float(data1.mean())
s2_mean = float(data2.mean())

# 方差未知但相等 均值之差的置信区间
# 推理公式 书上 P75-76
# 利用scipy 的 t f函数
loc_A = s1_mean - s2_mean
scale_A = math.sqrt((49 * s1_modified * s1_modified + 49 * s2_modified * s2_modified) / 98) * math.sqrt(
    1 / 48 + 1 / 48)

loc_B = 0
scale_B = (s1_modified * s1_modified) / (s2_modified * s2_modified)

print("均值差的95%置信区间：", stats.t.interval(0.95, 98, loc_A, scale_A))

print("方差比的95%置信区间：", stats.f.interval(0.95, 50 - 1, 50 - 1, loc_B, scale_B))

dataParzen = pandas.DataFrame(dataSet1)  # .sort_values(by=['mcv'])
# print(dataParzen.head(50))

# 方窗
dataParzen2 = pandas.DataFrame(dataSet['alkphos'].head(200))
dataList = dataParzen2['alkphos'].values.tolist()  # dataParzen['mcv'].values.tolist()
# dataList = dataParzen['mcv'].values.tolist()

print("begin ", len(dataList))

dataSorted = sorted(dataList)


# dataList.sort()
# print(dataList)

def parzenFang(x, h, N):
    datasort = sorted(x)
    b = 0
    a = []
    p = []

    for i in range(0, len(x)):
        for j in range(0, N):
            if abs((x[j] - datasort[i]) / h) <= 1 / 2:
                q = 1
            else:
                q = 0
            b = q + b

        a.append(b)
        b = 0

    for i in range(0, len(x)):
        p.append(1 / (N * h) * a[i])

    return p


def parzenGauss(x, h, N):
    datasort = sorted(x)
    b = 0
    h1 = h / math.sqrt(N)
    p = []

    for i in range(0, len(x)):
        for j in range(0, N):
            b = b + math.exp(((x[j] - datasort[i]) / h1) ** 2 / (-2)) / math.sqrt(2 * math.pi) / h1
        p.append(b / N)  # p[i] = b / N
        b = 0

    return p


p1 = parzenFang(dataList, 0.25, 100)
p2 = parzenFang(dataList, 4, 100)
p3 = parzenFang(dataList, 8, 100)

p4 = parzenGauss(dataList, 0.25, 200)
p5 = parzenGauss(dataList, 10, 200)
p6 = parzenGauss(dataList, 20, 200)

print(p1)
print(len(p1))
print(dataSorted)
print(len(dataSorted))

plt.subplot(2, 3, 1)
plot(dataSorted, p1)
plt.subplot(2, 3, 2)
plot(dataSorted, p2)
plt.subplot(2, 3, 3)
plot(dataSorted, p3)

plt.subplot(2, 3, 4)
plot(dataSorted, p4)
plt.subplot(2, 3, 5)
plot(dataSorted, p5)
plt.subplot(2, 3, 6)
plot(dataSorted, p6)

plt.show()


# kN 近邻
def KN(N, x):
    kn = math.sqrt(N)
    print('kn', kn)
    px = []
    y = []

    for i in range(0, len(x)):
        for j in range(0, N):
            y.append(abs(x[i] - x[j]))

        y = sorted(y)
        px.append(kn / N / (2 * y[i]))

    return px


def knn(X, kn, xleft, xright, xstep):
    k = 1
    px = []
    x = xleft
    while x < xright + xstep / 2:
        eucl = []
        for i in range(0, len(X)):
            eucl.append(math.sqrt(((abs((x - X[i]))) ** 2)))
        eucl = sorted(eucl)
        ro = eucl[kn]
        V = 2 * ro
        px.append(kn / (len(X) * V))
        k = k + 1
        x = x + xstep

    return px


print(max(dataList))
print(min(dataList))

k1 = knn(dataList, 10, min(dataList), max(dataList), (max(dataList) - min(dataList)) / len(dataList))  # KN(1, dataList)
k2 = knn(dataList, 50, min(dataList), max(dataList), (max(dataList) - min(dataList)) / len(dataList))  # KN(1, dataList)
k3 = knn(dataList, 100, min(dataList), max(dataList),(max(dataList) - min(dataList)) / len(dataList))  # KN(1, dataList)

k1.pop()
k2.pop()
k3.pop()

print(k1)
print(len(k1))

plot(dataSorted, k1, label='kn=10')
plot(dataSorted, k2, label='kn=50')
plot(dataSorted, k3, label='kn=100')
plt.legend()
plt.show()
