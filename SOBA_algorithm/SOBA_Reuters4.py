import numpy as np
from scipy import sparse
import scipy.io as sio

'''
Second Order Banditron Algorithm
Diagonal version
'''
X = sparse.load_npz("Reuters4_datasetX.npz")
Y = sparse.load_npz("Reuters4_datasetY.npz")
Y = Y + np.zeros(Y.shape)
y = np.array(Y)
y = y.flatten()
d = X.shape[0]  # number of features
K = 4  # number of classes
n = len(y)  # number of data
# initialization
a = 1  # regularization parameter
gamma = 2**(-7)  # exploration parameter
T = n  # number of rounds
W = np.zeros([K, d])
A = a * np.ones([K*d])
theta = np.zeros([K*d])
E = np.eye(K)
cumulative_margin = 0
z = np.zeros([K*d])
g = np.zeros([K*d])
correct = 0
print_fre = 2000
accu = np.zeros([T])
best_accuracy = 0
# SOBA algorithm
for t in range(T):
    x = np.array(X[:,t] + np.zeros(X[:,t].shape)).flatten()
    Wx = W.dot(x)
    yhat = np.argmax(Wx) + 1
    p = (1-gamma) * E[:, yhat-1] + gamma / K * np.ones([K])
    rand_num = np.random.random()
    for i in range(K):
        if sum(p[:i+1]) >= rand_num:
            ytilde = i+1
            break
    nt = 0
    if ytilde == y[t]:
        correct += 1
        Wx_bar = np.zeros([K-1])
        Wx_bar[:ytilde-1] = Wx[:ytilde-1]
        Wx_bar[ytilde-1:] = Wx[ytilde:]
        idx = np.argmax(Wx_bar)
        if idx <= ytilde - 2:
            ybar = idx + 1
        else:
            ybar = idx + 2
        g = 1 / p[ytilde-1] * np.kron(E[:, ybar-1]-E[:, ytilde-1], x)
        z = np.sqrt(p[ytilde-1]) * g
        #m = (sum(W.reshape(-1, 1)*z.reshape(-1, 1))**2 + 2*sum(W.reshape(-1, 1)*g.reshape(-1, 1))) / (1 + sum(z*z/A))
        m = ((W.reshape(1, -1).dot(z.reshape(-1, 1)))**2 + 2*W.reshape(1, -1).dot(g.reshape(-1, 1))) / (1 + z.reshape(1, -1).dot((z/A).reshape(-1, 1)))
        if cumulative_margin+m >= 0:
            nt = 1
        cumulative_margin += nt*m
    A += nt*z*z
    theta -= nt*g
    W = (theta/A).reshape(K, d)
    accu[t] = correct / (t+1)
    if t%print_fre == 0:
        print(t)
        print(correct/(t+1))
file_name = 'SOBA_accu_Reuters4_g_'+str(gamma)+'.mat'
sio.savemat(file_name,{'accu':accu})