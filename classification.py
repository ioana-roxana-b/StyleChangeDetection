import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import dataset_preprocessing
import supervised_models
import unsupervised_models

def classification(type, classifiers, data_df, preprocessing_methods = None, dialog=False):

    if dialog:
        X = data_df.drop('label', axis=1).values
        y = data_df['label']
    else:
        X = data_df.drop('label', axis=1).values
        y = data_df['label'].apply(lambda x: x.split()[0]).values

    skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    le = LabelEncoder()
    y_le = le.fit_transform(y)
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    if preprocessing_methods != None:
        for m in preprocessing_methods:
            dataset_preprocessing.apply_preprocessing(m, X_train, X_test, y_train)

    if type == 's':
        for c in classifiers:
            clf, y_pred = supervised_models.sup_models(X_train, y_train, X_test, c)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
            f1 = f1_score(y_test, y_pred, average='macro')

            print("Real values: ", y_test)
            print("Pred values: ", y_pred)

            results_df = pd.DataFrame({
                'Classifier': [c],
                'Preprocessing methods': [preprocessing_methods],
                'Accuracy': [accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'F1 Score': [f1]
            })
            results_df.to_csv(f'Outputs/results_{c}.csv', mode='a', index=False)
    elif type == 'u':
        print((np.unique(y)))
        for c in classifiers:
            if c == 'kmeans':
                clusters = unsupervised_models.unsup_models(X, c, len(np.unique(y)))
                unsupervised_models.visualize_clusters(c, X, clusters)
            elif c == 'pca':
                X_reduced = unsupervised_models.unsup_models(X, c)
                unsupervised_models.visualize_clusters(c, X_reduced, y_le)

    return 0
