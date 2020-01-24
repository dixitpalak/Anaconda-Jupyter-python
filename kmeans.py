import numpy as np
import re
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# ------------ Training data ------------
file = open("E:\\Python Projects\\AML\\arcane\\arcene_train.data", "r")
train_data = []

for line in file:
    r1 = re.split(r'\s', line)[5:7]
    r2 = [int(i) for i in r1]
    train_data.append(r2)

train_data_np = np.asarray(train_data, dtype=int)
file.close()
# ----------------------------------------

kmeans = KMeans(n_clusters=2, random_state=0).fit(train_data)

x = [i[0] for i in train_data]
y = [i[1] for i in train_data]
plt.scatter(x, y)
plt.show()

for i in range(len(train_data)):
    if kmeans.labels_[i] == 0:
        plt.scatter(x[i], y[i],c='red')
    else:
        plt.scatter(x[i], y[i], c='green')

plt.show()



