#usage: python3.10 PCA_example_2.py

"""
The Python version is 3.10.4.
This py script shows the procedure of PCA.

The script needs the following packages installed
numpy       1.21.6
sklearn     0.0
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

 
dataset = load_iris()
data = dataset.data

data_rows, data_columns = np.shape(data)
data_new = np.zeros(np.shape(data))

for i in range(data_columns):
    column_mean = np.mean(data[:,i])
    for j in range(data_rows):
        data_new[j][i] = data[j][i] - column_mean


#do by sklearn
pca = decomposition.PCA(n_components=4)
pca.fit(data_new)
X_new = pca.transform(data_new)
Vt = pca.components_
V_1 = Vt.T
print("The V by sklearn")
print(V_1)
singular_values = pca.singular_values_
print("Singular values by sklearn")
print(singular_values)
print('\n',end='')


"""
We get V^{T} when using sklearn.
see source code in
'/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/decomposition
_pca.py'

line 518-522

    U, S, Vt = linalg.svd(X, full_matrices=False)
    # flip eigenvectors' sign to enforce deterministic output
    U, Vt = svd_flip(U, Vt)

    components_ = Vt
"""

#do by hand
eigenvalues,eigenvectors = np.linalg.eigh(data_new.T @ data_new)

#sort the eigenvectors by eigenvalues in reverse order
idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

V_2 = np.linalg.inv(eigenvectors.T)
print("The V by hand")
print(V_2)
print("Sigular values by hand")
print(np.sqrt(eigenvalues))


