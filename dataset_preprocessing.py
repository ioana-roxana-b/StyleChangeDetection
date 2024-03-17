import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

def pca(X_train, X_test):
    pca = PCA(n_components=10)
    pca.fit(X_train)
    new_X = pca.transform(X_train)
    new_X_test = pca.transform(X_test)
    return new_X, new_X_test

def minmax_sc(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def stand_sc(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def lasso(X_train, X_test, y_train):
    lass = Lasso(alpha=0.01, max_iter=10000)
    lass.fit(X_train, y_train)
    coef = lass.coef_
    idx_nonzero = np.nonzero(coef)[0]
    X_train = X_train[:, idx_nonzero]
    X_test = X_test[:, idx_nonzero]
    return X_train, X_test

def lasso_threshold(X_train, X_test, y_train):
    threshold = 0.01
    lasso = Lasso(alpha=0.01, max_iter=100000, tol=1e-4)
    lasso.fit(X_train, y_train)
    coef = lasso.coef_

    idx_above_threshold = np.where(np.abs(coef) > threshold)[0]
    X_train = X_train[:, idx_above_threshold]
    X_test = X_test[:, idx_above_threshold]

    return X_train, X_test

def apply_preprocessing(method_name, X_train, X_test, y_train=None):
    preprocessing_functions = {
        'pca': lambda X_train, X_test, y_train=None: pca(X_train, X_test),
        'minmax_sc': lambda X_train, X_test, y_train=None: minmax_sc(X_train, X_test),
        'stand_sc': lambda X_train, X_test, y_train=None: stand_sc(X_train, X_test),
        'lasso': lambda X_train, X_test, y_train: lasso(X_train, X_test, y_train),
        'lasso_threshold': lambda X_train, X_test, y_train: lasso_threshold(X_train, X_test, y_train),
    }
    if method_name in preprocessing_functions:
        return preprocessing_functions[method_name](X_train, X_test, y_train)
    else:
        raise ValueError(f"Method {method_name} not recognized.")