## Author Kedong He
from sklearn.datasets import fetch_rcv1
from scipy import sparse
import numpy as np

''''
This file is to extract data which have exactly one label
from the set {CCAT, ECAT, GCAT, MCAT} from the original 
data set RCV1
'''
rcv1 = fetch_rcv1()
# Find the index of the label {CCAT, ECAT, GCAT, MCAT}
idx_ccat = np.argwhere(rcv1.target_names=='CCAT')[0,0]
idx_ecat = np.argwhere(rcv1.target_names=='ECAT')[0,0]
idx_gcat = np.argwhere(rcv1.target_names=='GCAT')[0,0]
idx_mcat = np.argwhere(rcv1.target_names=='MCAT')[0,0]
'''
Find the index of the data which have exactly one label
from the set {CCAT, ECAT, GCAT, MCAT}
'''
combination = rcv1.target[:, idx_ccat] + rcv1.target[:, idx_ecat] + rcv1.target[:, idx_gcat] + rcv1.target[:, idx_mcat]
idx = np.argwhere(combination == 1)[:, 0]
# Extract the data we need and save
X = rcv1.data[idx, :]
X = X.transpose()
sparse.save_npz("Reuters4_datasetX.npz", X)
# Transform the label to {1,2,3,4} and save
y = rcv1.target[idx, :]
y1 = 1 * (y[:, idx_ccat] == 1) + 2 * (y[:, idx_ecat] == 1) + 3 * (y[:, idx_gcat] == 1) + 4 * (y[:, idx_mcat] == 1)
sparse.save_npz("Reuters4_datasetY.npz", y1)
