import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import dataset_preprocessing
import supervised_models
import unsupervised_models

def classification(type, classifiers, data_df, preprocessing_methods = None, dialog=False):
    le = LabelEncoder()

    if dialog:
        X = data_df.drop('label', axis=1).values
        y = data_df['label']
    else:
        X = data_df.drop('label', axis=1).values
        y = data_df['label'].apply(lambda x: x.split()[0]).values

    y_le = le.fit_transform(y)
    labels = y_le

    if type != 'u':
        skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        if preprocessing_methods is not None:
            for m in preprocessing_methods:
                X_train, X_test = dataset_preprocessing.apply_preprocessing(m, X_train, X_test, y_train)

    else:  # Apply preprocessing to the entire dataset for unsupervised learning
        if preprocessing_methods is not None:
            for m in preprocessing_methods:
                X, _ = dataset_preprocessing.apply_preprocessing(m, X, y_train=y_le)

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
        #print((np.unique(y)))
        for c in classifiers:
            if c == 'kmeans':
                #unsupervised_models.elbow_method(X, max_clusters=30)  # Plot the elbow method
                best_num_clusters = unsupervised_models.grid_search_kmeans(X, min_clusters=2, max_clusters=20)  # Determine the best number of clusters
                clusters =  unsupervised_models.unsup_models(X, c, best_num_clusters)
                print(np.unique(clusters.labels_))
                print(np.unique(labels))
                unsupervised_models.evaluate_clustering(X, labels, clusters)
                unsupervised_models.visualize_clusters_v2(c, X, clusters, np.unique(y))
            elif c == 'pca':
                X_reduced = unsupervised_models.unsup_models(X, c)
                unsupervised_models.visualize_clusters(c, X_reduced, y_le)

    return 0