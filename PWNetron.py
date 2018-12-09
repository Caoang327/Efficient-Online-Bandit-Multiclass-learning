import numpy as np

X = np.loadtxt('SYNSEPdataX.dat')
Y = np.loadtxt('SYNSEPdataY.dat')
def predict_label(W,x):
    out = np.dot(W,x)
    return np.argmax(out)+1
def compute_P(W,x):
    p = np.dot(W,x)
    e_p = np.exp(p - np.max(p))
    soft_max = e_p / e_p.sum()
    return soft_max

gamma = 0.01
alpha = 10
betta = 0.01
k = 9
d = 400
D = 1
correct = 0
t = 0
W = np.zeros([k,d])
W_slid = W
np.random.seed(0)
A_accu = 1/D
bt = 0
counter = 0
accu = []
print_fre = 100

for i in range(X.shape[1]):
    counter = counter + 1
    x = X[:,i].reshape(-1,1)
    y = int(Y[i])
    pt = compute_P(W_slid,x)
    pt_silde = (1-gamma)*pt + gamma/k
    if np.random.random() >= gamma:
        W = W_slid
        y_hat = predict_label(W,x)
    else:
        W = np.zeros([k,d])
        roll = np.random.randint(1,k+1)
        y_hat = roll
    if y_hat == y:
        eyt = np.zeros([k,1])
        eyt[y-1][0] = 1
        delta = (1 - pt[y-1,0])/(pt_silde[y-1,0])*np.kron((1/k-eyt),x)
        kt = pt_silde[y-1,0]
        correct += 1
    else:
        eyt = np.zeros([k,1])
        eyt[y_hat-1,0] = 1
        delta = (pt[y_hat-1,0]/pt_silde[y_hat-1,0])*np.kron(eyt - 1/k,x)
        kt = 1
    A_accu = A_accu + kt*betta*(delta**2)
    W_T = np.transpose(W)
    W_Slack = W_T.reshape(-1,1)
    bt = bt + (1 - kt*betta*np.dot(delta.reshape(1,-1),W_Slack))*delta
    W_slid = -np.dot(np.diagflat(1/A_accu),bt)
    W_slid = W_slid.reshape([k,-1])
    accu.append(correct*1.0/counter)
    if counter%print_fre ==1:
        print(correct*1.0/counter)