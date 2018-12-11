import numpy as np
import scipy.io as sio
from scipy import sparse
def predict_label(W,x):
    out = np.dot(W,x)
    return np.argmax(out)+1


X = sparse.load_npz("Reuters4_datasetX.npz")
Y = sparse.load_npz("Reuters4_datasetY.npz")
Y = Y + np.zeros(Y.shape)
y = np.array(Y)
y = y.flatten()
X = X + np.zeros(X.shape)
X = np.array(X)
d = X.shape[0]
k = 4
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
file_name = 'Perceptron_Reusters_accu.mat'
sio.savemat(file_name,{'accu':accu})