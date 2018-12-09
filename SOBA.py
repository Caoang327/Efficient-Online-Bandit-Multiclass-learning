import numpy as np

'''
generate data set SYNSEP
a 9-class, 400-dimensional synthetic data set of size 10^6
'''
d = 400  # number of features
K = 9  # number of classes
n = 10**6  # number of data
# generate the fixed 9 bit-vectors
V = np.zeros([d, K])
idx_support = np.zeros([120, 9], int)
Nsupport = np.zeros([9], int)
for i in range(K):
    idx_permuted = np.random.permutation(range(120))
    idx_support[:, i] = idx_permuted
    Nsupport[i] = np.random.randint(20, 41)
    V[idx_permuted[:Nsupport[i]], i] = np.ones([Nsupport[i]])
# generate the data set
X = np.zeros([d, n])
for i in range(n):
    j = np.random.randint(0, 9)
    X[:, i] = V[:, j]
    idx_turnoff = np.random.permutation(idx_support[:Nsupport[j], j])[:5]
    idx_turnon = np.random.permutation(range(120, 400))[:20]
    X[idx_turnoff, i] = np.zeros([5])
    X[idx_turnon, i] = np.ones([20])

print(sum(V[:120]))
print(sum(V[120:]))
print(X[:, 0])
