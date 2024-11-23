import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso

def pca(X_train, X_test=None, **params):
    """
    Applies Principal Component Analysis (PCA) to reduce the dimensionality of the dataset.

    Params:
        X_train (array-like): Training data.
        X_test (array-like, optional): Test data to transform using the same PCA model.
        **params: Additional parameters for PCA (e.g., 'n_components').

    Returns:
        tuple: Transformed training data, and transformed test data (if provided).
    """

    n_components = params.get('n_components', 5)  # Default to 5 if not specified
    pca_model = PCA(n_components=n_components)
    X_train = pca_model.fit_transform(X_train)

    if X_test is not None:
        X_test = pca_model.transform(X_test)

    return X_train, X_test

def minmax_sc(X_train, X_test=None, **params):
    """
    Applies Min-Max Scaling to normalize the feature values to the range [0, 1].

    Params:
        X_train (array-like): Training data.
        X_test (array-like, optional): Test data to transform using the same scaler.
        **params: Additional parameters for MinMaxScaler (e.g., 'feature_range').

    Returns:
        tuple: Scaled training data, and scaled test data (if provided).
    """
    feature_range = params.get('feature_range', (0, 1))  # Default to (0, 1)
    scaler = MinMaxScaler(feature_range=feature_range)
    X_train = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test = scaler.transform(X_test)

    return X_train, X_test

def stand_sc(X_train, X_test=None, **params):
    """
    Applies Standard Scaling to standardize the features to have a mean of 0 and variance of 1.

    Params:
        X_train (array-like): Training data.
        X_test (array-like, optional): Test data to transform using the same scaler.
        **params: Additional parameters for StandardScaler (currently none are used).

    Returns:
        tuple: Standardized training data, and standardized test data (if provided).
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test = scaler.transform(X_test)

    return X_train, X_test

def lasso(X_train, X_test=None, y_train=None, **params):
    """
    Applies Lasso regression for feature selection by removing features with zero coefficients.

    Params:
        X_train (array-like): Training data.
        X_test (array-like, optional): Test data to transform using the selected features.
        y_train (array-like): Target values for training data.
        **params: Additional parameters for Lasso (e.g., 'alpha').

    Returns:
        tuple: Transformed training data, and transformed test data (if provided).
    """
    alpha = params.get('alpha', 0.1)  # Default alpha to 0.1
    max_iter = params.get('max_iter', 10000)  # Default max iterations to 10000

    lasso_model = Lasso(alpha=alpha, max_iter=max_iter)
    lasso_model.fit(X_train, y_train)

    coef = lasso_model.coef_
    idx_nonzero = np.nonzero(coef)[0]  # Indices of non-zero coefficients

    X_train = X_train[:, idx_nonzero]

    if X_test is not None:
        X_test = X_test[:, idx_nonzero]

    return X_train, X_test

def lasso_threshold(X_train, X_test=None, y_train=None, **params):
    """
    Applies Lasso regression for feature selection based on a coefficient threshold.

    Params:
        X_train (array-like): Training data.
        X_test (array-like, optional): Test data to transform using the selected features.
        y_train (array-like): Target values for training data.
        **params: Additional parameters for Lasso, including 'threshold' and 'alpha'.

    Returns:
        tuple: Transformed training data, and transformed test data (if provided).
    """
    threshold = params.get('threshold', 0.1)  # Default threshold to 0.1
    alpha = params.get('alpha', 0.01)  # Default alpha to 0.01
    max_iter = params.get('max_iter', 10000)  # Default max iterations to 10000

    lasso_model = Lasso(alpha=alpha, max_iter=max_iter)
    lasso_model.fit(X_train, y_train)

    coef = lasso_model.coef_
    idx_above_threshold = np.where(np.abs(coef) > threshold)[0]  # Indices above threshold

    X_train = X_train[:, idx_above_threshold]

    if X_test is not None:
        X_test = X_test[:, idx_above_threshold]

    return X_train, X_test

def apply_preprocessing(method_name, X_train, X_test=None, y_train=None, **params):
    """
    Applies a specified preprocessing or feature selection method to the data.

    Params:
        method_name (str): The name of the method to apply.
        X_train (array-like): Training data.
        X_test (array-like, optional): Test data (if applicable).
        y_train (array-like, optional): Target values for training data (if applicable).
        **params: Additional parameters to pass to the preprocessing method.

    Returns:
        tuple: Transformed training data, and transformed test data (if provided).
    """
    # Dictionary to map method names to their respective functions
    preprocessing_functions = {
        'pca': lambda: pca(X_train, X_test, **params),
        'minmax_sc': lambda: minmax_sc(X_train, X_test, **params),
        'stand_sc': lambda: stand_sc(X_train, X_test, **params),
        'lasso': lambda: lasso(X_train, X_test, y_train, **params),
        'lasso_threshold': lambda: lasso_threshold(X_train, X_test, y_train, **params),
    }

    # Apply the selected method if it exists in the dictionary
    if method_name in preprocessing_functions:
        return preprocessing_functions[method_name]()
    else:
        raise ValueError(f"Method {method_name} not recognized.")
