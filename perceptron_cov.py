import numpy as np
import scipy.io as sio
X = np.loadtxt('CovtypedataX.dat')
Y = np.loadtxt('CovtypedataY.dat')
def predict_label(W,x):
    out = np.dot(W,x)
    return np.argmax(out)+1
k = 7
d = X.shape[0]
W = np.zeros([k,d])
n = X.shape[1]
T = n
accu = np.zeros(n)
correct = 0
for t in range(T):
    x = X[:,t].reshape(-1,1)
    y = int(Y[t])
    y_hat = predict_label(W,x)
    if y_hat != y:
        for j in range(W.shape[0]):
            if (j+1) == y:
                tao = 1
            elif (j+1) == y_hat:
                tao = -1
            else:
                tao = 0
            W[j,:] = W[j,:]+x.flatten()*tao
    elif y_hat == y:
        correct = correct + 1
    accu[t] = correct/(t+1)
file_name = 'Perceptron_Cov_accu.mat'
sio.savemat(file_name,{'accu':accu})