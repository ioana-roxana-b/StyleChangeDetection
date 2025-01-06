from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV


def random_forest(X_train, y_train, X_test, **params):
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return clf, y_pred_proba


def grad_boost(X_train, y_train, X_test, **params):
    clf = GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return clf, y_pred_proba


def lightGBM(X_train, y_train, X_test, **params):
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return clf, y_pred_proba


def svm(X_train, y_train, X_test, **params):
    clf = SVC(probability=True, cache_size=1024, **params)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return clf, y_pred_proba


def knn(X_train, y_train, X_test, **params):
    clf = KNeighborsClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return clf, y_pred_proba


def naive_bayes(X_train, y_train, X_test, **params):
    clf = GaussianNB(**params)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return clf, y_pred_proba


def voting(X_train, y_train, X_test, voting='soft', clf1=None, clf2=None, clf3=None, clf4=None, clf5=None, clf6=None):
    estimators = []
    if clf1 is not None:
        estimators.append(('rf', clf1))
    if clf2 is not None:
        estimators.append(('gb', clf2))
    if clf3 is not None:
        estimators.append(('lgbm', clf3))
    if clf4 is not None:
        estimators.append(('knn', clf4))
    if clf5 is not None:
        estimators.append(('nb', clf5))
    if clf6 is not None:
        estimators.append(('svm', clf6))

    voting_clf = VotingClassifier(estimators=estimators, voting=voting)
    voting_clf.fit(X_train, y_train)
    y_pred_proba = voting_clf.predict_proba(X_test)[:, 1]
    return voting_clf, y_pred_proba


def get_voting_classifier(X_train, y_train, X_test):
    clf1, _ = random_forest(X_train, y_train, X_test)
    clf2, _ = grad_boost(X_train, y_train, X_test)
    clf3, _ = lightGBM(X_train, y_train, X_test)
    clf4, _ = knn(X_train, y_train, X_test)
    clf5, _ = naive_bayes(X_train, y_train, X_test)
    clf6, _ = svm(X_train, y_train, X_test)

    clf, y_pred_proba = voting(X_train, y_train, X_test, clf1=clf1, clf2=clf2, clf3=clf3, clf4=clf4, clf5=clf5, clf6=clf6)
    return clf, y_pred_proba


def sup_models(X_train, y_train, X_test, c=None, **params):
    """
    Applies a specified supervised learning model to the given training and test data.

    Params:
        X_train (array-like): Training feature matrix.
        y_train (array-like): Training target labels.
        X_test (array-like): Test feature matrix.
        c (str, optional): Name of the supervised learning model to apply. Must be one of:
                           'random_forest', 'grad_boost', 'lightGBM', 'knn', 'naive_bayes', 'svm', or 'voting'.
        **params: Parameters for the specified classifier.

    Returns:
        tuple:
            clf (model object): The trained classifier.
            y_pred_proba (array-like): Predicted probabilities for the test data.
    """
    model_functions = {
        'random_forest': lambda: random_forest(X_train, y_train, X_test, **params),
        'grad_boost': lambda: grad_boost(X_train, y_train, X_test, **params),
        'lightGBM': lambda: lightGBM(X_train, y_train, X_test, **params),
        'knn': lambda: knn(X_train, y_train, X_test, **params),
        'naive_bayes': lambda: naive_bayes(X_train, y_train, X_test, **params),
        'svm': lambda: svm(X_train, y_train, X_test, **params),
        'voting': lambda: get_voting_classifier(X_train, y_train, X_test)
    }

    if c not in model_functions:
        raise ValueError(f"Classifier '{c}' is not recognized.")

    clf, y_pred_proba = model_functions[c]()

    return clf, y_pred_proba
