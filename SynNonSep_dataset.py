## Author Hangting Cao
import numpy as np

'''
generate data set SYNNONSEP
a 9-class, 400-dimensional synthetic data set of size 10^6
X and y are the same as data set SYNSEP 
introduce y_ob with 5% label noise
'''
K = 9
y = np.loadtxt('SYNSEPdataY.dat')
n = len(y)
# induce 5% label noise
idx_noise = np.random.permutation(range(n))[:int(0.05*n)]
label_latent = np.zeros([K-1])
y_ob = np.copy(y)

for idx in idx_noise:
    label_latent[:int(y[idx])-1] = range(1, int(y[idx]))
    label_latent[int(y[idx])-1:] = range(int(y[idx])+1, 10)
    rand_num = np.random.randint(0, 8)
    y_ob[idx] = label_latent[rand_num]
np.savetxt('SYNNONSEPdataYob.dat', y_ob)

