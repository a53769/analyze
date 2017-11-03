

# Begin

#    1. sepal length in cm (花萼长)
#    2. sepal width in cm（花萼宽）
#    3. petal length in cm (花瓣长)
#    4. petal width in cm（花瓣宽）

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import statistics
from pylab import plot

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

from numpy import genfromtxt, zeros
localfn='D://workspace/analyze/iris.csv'
# data = genfromtxt(localfn,delimiter=',',usecols=(0,1,2,3,4))
dataSet = pandas.read_csv(localfn, names=['s-length', 's-width', 'p-length', 'p-width', 'type'])

print(dataSet.describe())
print("--- --- ---")


def RandomSampling(data):
    sample = []
    for i in range(1,1000):
        temp = data.sample(frac=0.3, replace=False).mean()
        sample.append(temp)
    return sample

def normfun(x,mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

def Gauss(typeName):
    dataSet1 = dataSet[dataSet['type'] == typeName]
    fl = dataSet1['s-length']
    mean = RandomSampling(fl)
    # sns.distplot(mean,bins=20, rug=True, fit=stats.gamma, norm_hist=False)

    plt.hist(mean, bins=20, color='r', alpha=0.5, rwidth=1, normed=False,histtype='stepfilled')
    plt.legend(loc='best', frameon=False)
    # mu = fl.mean()
    # sigma = fl.std()
    # print(typeName, " s-length 均值和方差：")
    # print(mu)
    # print(sigma)
    # x = np.arange(4, 6, 0.05)
    # y = normfun(x, mu, sigma)
    # x = np.linspace(mu - 5 * sigma, mu + 5 * sigma)
    # y = stats.norm.pdf(x, mu, sigma)
    # plt.plot(x, y, color='blue')
    # plt.hist(mean, bins=10, color='r', alpha=0.5, rwidth=0.5, normed=True)

    # plt.title('Gauss: $\mu$=%.2f, $\sigma^2$=%.2f' % (mu, sigma))
    # plt.xlabel('sepal length')
    # plt.ylabel('Probability density')
    plt.show()

# Gauss("setosa")

dataSet1 = dataSet[dataSet['type'] == 'setosa']
df1 = pandas.DataFrame(dataSet1)
dataSet2 = dataSet[dataSet['type'] == 'versicolor']  # Iris-virginica
df2 = pandas.DataFrame(dataSet2)
dataSet3 = dataSet[dataSet['type'] == 'virginica']
df3 = pandas.DataFrame(dataSet3)

df = pandas.DataFrame(dataSet)



# 盒子图
df1.plot.box()
plt.show()


# 三种亚种比较
plt.subplot(2, 2, 1)
plot(df1['s-length'], df1['s-width'], 'b+', label='setosa')
plot(df2['s-length'], df2['s-width'], 'r+', label='versicolor')
plot(df3['s-length'], df3['s-width'], 'g+', label='virginica')
plt.legend(loc='upper left')
# plt.title("sepal length & sepal width")  # petal
plt.xlabel('sepal length')
plt.ylabel('sepal width')
# show()

plt.subplot(2, 2, 2)
plot(df1['s-length'], df1['p-length'], 'b+', label='setosa')
plot(df2['s-length'], df2['p-length'], 'r+', label='versicolor')
plot(df3['s-length'], df3['p-length'], 'g+', label='virginica')
plt.legend(loc='upper left')
# plt.title("sepal length & petal length")  # petal
plt.xlabel('sepal length')
plt.ylabel('petal length')
# show()

plt.subplot(2, 2, 3)
plot(df1['s-length'], df1['p-width'], 'b+', label='setosa')
plot(df2['s-length'], df2['p-width'], 'r+', label='versicolor')
plot(df3['s-length'], df3['p-width'], 'g+', label='virginica')
plt.legend(loc='upper left')
# plt.title("sepal length & petal width")  # petal
plt.xlabel('sepal length')
plt.ylabel('petal width')
# show()

plt.subplot(2, 2, 4)
plot(df1['p-length'], df1['p-width'], 'b+', label='setosa')
plot(df2['p-length'], df2['p-width'], 'r+', label='versicolor')
plot(df3['p-length'], df3['p-width'], 'g+', label='virginica')
plt.legend(loc='upper left')
# plt.title("petal length & petal width")  # petal
plt.xlabel('petal length')
plt.ylabel('petal width')
# show()

plt.show()


from pandas.plotting import autocorrelation_plot
auto = pandas.Series(dataSet1['s-length'])
autocorrelation_plot(auto)
plt.show()


from pandas.plotting import radviz
radviz(dataSet,'type')
plt.show()



sns.pairplot(dataSet, hue='type', size=3, diag_kind='kde')
plt.show()
