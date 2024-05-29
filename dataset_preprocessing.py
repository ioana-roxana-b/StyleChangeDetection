import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

def pca(X_train, X_test=None):
    pca = PCA(n_components=4)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    if X_test is not None:
        X_test = pca.transform(X_test)
    return X_train, X_test

def minmax_sc(X_train, X_test=None):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)
    return X_train, X_test

def stand_sc(X_train, X_test=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)
    return X_train, X_test

def lasso(X_train, X_test=None, y_train=None):
    lass = Lasso(alpha=0.01, max_iter=10000)
    lass.fit(X_train, y_train)
    coef = lass.coef_
    idx_nonzero = np.nonzero(coef)[0]
    X_train = X_train[:, idx_nonzero]
    if X_test is not None:
        X_test = X_test[:, idx_nonzero]
    return X_train, X_test

def lasso_threshold(X_train, X_test=None, y_train=None):
    threshold = 0.01
    lasso = Lasso(alpha=0.01, max_iter=100000, tol=1e-4)
    lasso.fit(X_train, y_train)
    coef = lasso.coef_

    idx_above_threshold = np.where(np.abs(coef) > threshold)[0]
    X_train = X_train[:, idx_above_threshold]
    if X_test is not None:
        X_test = X_test[:, idx_above_threshold]
    return X_train, X_test

def apply_preprocessing(method_name, X_train, X_test=None, y_train=None):
    preprocessing_functions = {
        'pca': lambda: pca(X_train, X_test),
        'minmax_sc': lambda: minmax_sc(X_train, X_test),
        'stand_sc': lambda: stand_sc(X_train, X_test),
        'lasso': lambda: lasso(X_train, X_test, y_train),
        'lasso_threshold': lambda: lasso_threshold(X_train, X_test, y_train),
    }
    if method_name in preprocessing_functions:
        return preprocessing_functions[method_name]()
    else:
        raise ValueError(f"Method {method_name} not recognized.")
