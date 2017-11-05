import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


localfn='D:\\workspace\\analyze\\bupa.csv'
dataSet = pandas.read_csv(localfn, names=['mcv', 'alkphos', 'sgpt', 'sgot','gammagt','drinks', 'selector'])

def RandomSampleMean(data):
    sample = []
    for i in range(1,50):
        temp = data.sample(frac=0.1, replace=False).mean()
        sample.append(temp)
    return sample

def RandomSampleStd(data):
    sample = []
    for i in range(1,50):
        temp = data.sample(frac=0.1, replace=False).std()
        sample.append(temp)
    return sample

def RandomSampleStdCompare(data1,data2):
    sample = []
    for i in range(1, 50):
        temp1 = data1.sample(frac=0.1, replace=False).std()
        temp2 = data2.sample(frac=0.1, replace= False).std()
        sample.append(temp1/temp2)
    return sample



dataSet1 = dataSet[dataSet['selector'] == 1]
dataSet2 = dataSet[dataSet['selector'] == 2]
f1 = dataSet1['mcv']
f2 = dataSet2['mcv']

plt.figure(1)
plt.subplot(2,1,1)
data = RandomSampleMean(f1)
sns.distplot(data, bins=20, rug=True, fit=stats.gamma, norm_hist=False)

plt.subplot(2,1,2)
data2 = RandomSampleMean(f2)
sns.distplot(data2, bins=20, rug=True, fit=stats.gamma, norm_hist=False)


# plt.hist(mean, bins=10, color='r', alpha=0.5, rwidth=0.9, normed=True)
# mean, std = stats.norm.fit(data)
# print(mean)
# print(std)



plt.figure(2)

plt.subplot(2,1,1)
data = RandomSampleStd(f1)
sns.distplot(data, bins=20, rug=True,norm_hist=False)

plt.subplot(2,1,2)
data2 = RandomSampleStd(f1)
sns.distplot(data2, bins=20, rug=True, norm_hist=False)


plt.figure(3)

data = RandomSampleStdCompare(f1, f2)
sns.distplot(data2, bins=20, rug=True, norm_hist=False)

plt.show()