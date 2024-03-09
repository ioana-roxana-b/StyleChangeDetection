import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import dataset_preprocessing
import supervised_models
def classification(c, data_df, pc=False,
                  scal=False, minmax=False, lasso=False, lasso_t=False, rfe=False):

    X = data_df.drop('label', axis=1).values
    y = data_df['label'].apply(lambda x: x.split()[0]).values

    skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    #print(y_test)
    y_test = le.transform(y_test)

    #Se aplica metodele de preprocesare a setului de trăsături
    if minmax == True:
        X_train, X_test = dataset_preprocessing.minmax_sc(X_train, X_test)

    if scal == True:
        X_train, X_test = dataset_preprocessing.stand_sc(X_train, X_test)

    if lasso == True:
        X_train, X_test = dataset_preprocessing.lasso(X_train, X_test, y_train)

    if lasso_t == True:
        X_train, X_test = dataset_preprocessing.lasso_threshold(X_train, X_test, y_train)

    if rfe:
        X_train, rfe_selector = dataset_preprocessing.recursive_feature_elimination(X_train, y_train)
        X_test = rfe_selector.transform(X_test)

    if pc == True:
        X_train, X_test = dataset_preprocessing.pca(X_train, X_test)

    clf, y_pred, clf_name = supervised_models.pick(X_train, y_train, X_test, c)


    #Se calculează metricile de performanță și se salvează într-un fișier csv
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro')


    print("f=0, s=1")
    print("Real values: ", y_test)
    print("Pred values: ", y_pred)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)

    results_df = pd.DataFrame({
        'Classifier': [clf_name],
        'Configuration': [ f'pca={pc}, scal={scal}, minmax={minmax}, lasso={lasso}, lasso_threshold={lasso_t}'],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    })
    results_df.to_csv(f'results_{clf_name}.csv', mode='a', index=False)
    return accuracy
