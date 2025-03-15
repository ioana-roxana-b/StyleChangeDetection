from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

def random_forest(X_train, y_train, X_test, **params):
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

def grad_boost(X_train, y_train, X_test, **params):
    clf = GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

def lightGBM(X_train, y_train, X_test, **params):
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

def svm(X_train, y_train, X_test, **params):
    #tuned_parameters = [{'kernel': ['linear'], 'C': [1]}]
    #clf = GridSearchCV(SVC(probability=False, cache_size = 1024, degree = 2), tuned_parameters, scoring = 'accuracy')
    clf = SVC(probability=False, cache_size=1024, **params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

def knn(X_train, y_train, X_test, **params):
    clf = KNeighborsClassifier( **params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

def naive_bayes(X_train, y_train, X_test, **params):
    clf = GaussianNB( **params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred


def voting(X_train, y_train, X_test, voting = 'soft', clf1 = None, clf2 = None, clf3 = None, clf4 = None, clf5 = None, clf6 = None):
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

    voting_clf = VotingClassifier(estimators = estimators, voting = voting)
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    return voting_clf, y_pred


def get_voting_classifier(X_train, y_train, X_test):

    clf1, _ = random_forest(X_train, y_train, X_test)
    clf2, _ = grad_boost(X_train, y_train, X_test)
    clf3, _ = lightGBM(X_train, y_train, X_test)
    clf4, _ = knn(X_train, y_train, X_test)
    clf5, _ = naive_bayes(X_train, y_train, X_test)
    clf6, _ = svm(X_train, y_train, X_test)

    clf, y_pred = voting(X_train, y_train, X_test, clf1, clf2, clf3, clf4, clf5, clf6)
    return clf, y_pred

def sup_models(X_train, y_train, X_test, c = None, **params):
    """
    Applies a specified supervised learning model to the given training and test_scripts data.

    Params:
        X_train (array-like): Training feature matrix.
        y_train (array-like): Training target labels.
        X_test (array-like): Test feature matrix.
        c (str, optional): Name of the supervised learning model to apply. Must be one of the following:
                           'random_forest', 'grad_boost', 'lightGBM', 'knn', 'naive_bayes', 'svm', or 'voting'.
        **params: Parameters for the specified classifier.

    Returns:
        tuple:
            clf (model object): The trained classifier.
            y_pred (array-like): Predicted labels for the test_scripts data.
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

    clf, y_pred = model_functions[c]()

    return clf, y_pred

