import numpy as np
import scipy.io as sio

'''
Second Order Banditron Algorithm
Diagonal version
'''
X = np.loadtxt('SYNSEPdataX.dat')
y_true = np.loadtxt('SYNSEPdataY.dat')
y = np.loadtxt('SYNNONSEPdataYob.dat')
d = 400  # number of features
K = 9  # number of classes
n = len(y)  # number of data
# initialization
a = 1  # regularization parameter
gamma = 2**(-7) # exploration parameter
T = n  # number of rounds
W = np.zeros([K, d])
A = a * np.ones([K*d])
theta = np.zeros([K*d])
E = np.eye(K)
cumulative_margin = 0
z = np.zeros([K*d])
g = np.zeros([K*d])
correct = 0
print_fre = 1000
accu = np.zeros([T])
# SOBA algorithm
for t in range(T):
    x = X[:,t]
    Wx = W.dot(x)
    yhat = np.argmax(Wx) + 1
    p = (1-gamma) * E[:, yhat-1] + gamma / K * np.ones([K])
    rand_num = np.random.random()
    for i in range(K):
        if sum(p[:i+1]) >= rand_num:
            ytilde = i+1
            break
    if ytilde == y_true[t]:
        correct += 1
    nt = 0
    if ytilde == y[t]:
        Wx_bar = np.zeros([K-1])
        Wx_bar[:ytilde-1] = Wx[:ytilde-1]
        Wx_bar[ytilde-1:] = Wx[ytilde:]
        ybar = np.argmax(Wx)+1
        g = 1 / p[ytilde-1] * np.kron(E[:, ybar-1]-E[:, ytilde-1], x)
        z = np.sqrt(p[ytilde-1]) * g
        m = (sum(W.reshape(-1, 1)*z.reshape(-1, 1))**2 + 2*sum(W.reshape(-1, 1)*g.reshape(-1, 1))) / (1 + sum(z*z/A))
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
file_name = 'SOBA1_accu_sysnonsep_g_'+str(gamma)+'.mat'
sio.savemat(file_name,{'accu':accu})