import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

def pca(X_train, X_test = None):
    """
    Applies Principal Component Analysis (PCA) to reduce the dimensionality of the dataset.
    Params:
        X_train (array-like): Training data.
        X_test (array-like, optional): Test data to transform using the same PCA model.
    Returns:
        tuple: Transformed training data, and transformed test_scripts data (if provided).
    """
    pca = PCA(n_components = 5)
    pca.fit(X_train)
    X_train = pca.transform(X_train)

    if X_test is not None:
        X_test = pca.transform(X_test)

    return X_train, X_test

def minmax_sc(X_train, X_test = None):
    """
    Applies Min-Max Scaling to normalize the feature values to the range [0, 1].
    Params:
        X_train (array-like): Training data.
        X_test (array-like, optional): Test data to transform using the same scaler.
    Returns:
        tuple: Scaled training data, and scaled test_scripts data (if provided).
    """
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)

    return X_train, X_test

def stand_sc(X_train, X_test = None):
    """
    Applies Standard Scaling to standardize the features_methods to have a mean of 0 and variance of 1.
    Params:
        X_train (array-like): Training data.
        X_test (array-like, optional): Test data to transform using the same scaler.
    Returns:
        tuple: Standardized training data, and standardized test_scripts data (if provided).
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)

    return X_train, X_test

def lasso(X_train, X_test = None, y_train = None):
    """
    Applies Lasso regression for feature selection by removing features_methods with zero coefficients.
    Params:
        X_train (array-like): Training data.
        X_test (array-like, optional): Test data to transform using the selected features_methods.
        y_train (array-like): Target values for training data.
    Returns:
        tuple: Transformed training data, and transformed test_scripts data (if provided).
    """
    lass = Lasso(alpha=0.1, max_iter=10000)
    lass.fit(X_train, y_train)
    coef = lass.coef_
    idx_nonzero = np.nonzero(coef)[0]

    # Select features_methods with non-zero coefficients
    X_train = X_train[:, idx_nonzero]
    if X_test is not None:
        X_test = X_test[:, idx_nonzero]

    return X_train, X_test

def lasso_threshold(X_train, X_test = None, y_train = None):
    """
    Applies Lasso regression for feature selection based on a coefficient threshold.
    Params:
        X_train (array-like): Training data.
        X_test (array-like, optional): Test data to transform using the selected features_methods.
        y_train (array-like): Target values for training data.
    Returns:
        tuple: Transformed training data, and transformed test_scripts data (if provided).
    """
    threshold = 0.1
    lasso = Lasso(alpha=0.01, max_iter=10000, tol=1e-4)
    lasso.fit(X_train, y_train)
    coef = lasso.coef_

    # Select indices of features_methods with coefficients above the threshold
    idx_above_threshold = np.where(np.abs(coef) > threshold)[0]
    X_train = X_train[:, idx_above_threshold]

    if X_test is not None:
        X_test = X_test[:, idx_above_threshold]

    return X_train, X_test

def apply_preprocessing(method_name, X_train, X_test = None, y_train = None):
    """
    Applies a specified preprocessing or feature selection method to the data.
    Params:
        method_name (str): The name of the method to apply.
        X_train (array-like): Training data.
        X_test (array-like, optional): Test data (if applicable).
        y_train (array-like, optional): Target values for training data (if applicable).
    Returns:
        tuple: Transformed training data, and transformed test_scripts data (if provided).
    """

    # Dictionary to map method names to their respective functions
    preprocessing_functions = {
        'pca': lambda: pca(X_train, X_test),
        'minmax_sc': lambda: minmax_sc(X_train, X_test),
        'stand_sc': lambda: stand_sc(X_train, X_test),
        'lasso': lambda: lasso(X_train, X_test, y_train),
        'lasso_threshold': lambda: lasso_threshold(X_train, X_test, y_train),
    }

    # Apply the selected method if it exists in the dictionary
    if method_name in preprocessing_functions:
        return preprocessing_functions[method_name]()

    else:
        raise ValueError(f"Method {method_name} not recognized.")
