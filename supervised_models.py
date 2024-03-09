from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

def random_forest(X_train, y_train, X_test):
    clf = RandomForestClassifier(n_estimators=1000, random_state=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

def grad_boost(X_train, y_train, X_test):
    clf = GradientBoostingClassifier(n_estimators=1000, random_state=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

def lightGBM(X_train, y_train, X_test):
    clf = lgb.LGBMClassifier(n_estimators=500, random_state=50)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

def svm(X_train, y_train, X_test):
    tuned_parameters = [{'kernel': ['linear'], 'C': [1]}]
    clf = GridSearchCV(SVC(probability=False, cache_size=1024, degree=2), tuned_parameters, scoring='accuracy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

def knn(X_train, y_train, X_test, n_neighbors=5):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

def naive_bayes(X_train, y_train, X_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

def voting(X_train, y_train, X_test, clf1=None, clf2=None, clf3=None, clf4=None, clf5=None, clf6=None):
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', clf1),
            ('gb', clf2),
            ('lgbm', clf3),
            ('knn', clf4),
            ('nb', clf5),
            ('svm', clf6)
        ],
        voting='hard'
    )

    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    return voting_clf, y_pred

def pick(X_train, y_train, X_test, c=0):
    if c == 1:
        clf, y_pred = random_forest(X_train, y_train, X_test)
    elif c == 2:
        clf, y_pred = grad_boost(X_train,y_train,X_test)
    elif c == 3:
        clf, y_pred = lightGBM(X_train, y_train, X_test)
    elif c == 4:
        clf, y_pred = knn(X_train, y_train, X_test)
    elif c == 5:
        clf, y_pred = naive_bayes(X_train, y_train, X_test)
    elif c == 6:
        clf, y_pred = svm(X_train, y_train, X_test)
    elif c == 7:
        clf1, y_pred = random_forest(X_train, y_train, X_test)
        clf2, y_pred = grad_boost(X_train, y_train, X_test)
        clf3, y_pred = lightGBM(X_train, y_train, X_test)
        clf4, y_pred = knn(X_train, y_train, X_test)
        clf5, y_pred = naive_bayes(X_train, y_train, X_test)
        clf6, y_pred = svm(X_train, y_train, X_test)
        clf, y_pred = voting(X_train, y_train, X_test, clf1=clf1, clf2=clf2, clf3=clf3, clf4=clf4, clf5=clf5, clf6=clf6)

    clf_name = clf.__class__.__name__

    return clf, y_pred, clf_name


