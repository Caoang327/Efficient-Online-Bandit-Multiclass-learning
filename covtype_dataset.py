from sklearn.datasets import fetch_covtype
import numpy as np
covtype = fetch_covtype()
X = covtype.data.transpose()
y = covtype.target
np.savetxt('CovtypedataX.dat', X)
np.savetxt('CovtypedataY.dat', y)