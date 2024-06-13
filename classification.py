import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import dataset_preprocessing
import supervised_models
import unsupervised_models

def classification(type, classifiers, data_df, preprocessing_methods=None, dialog=False):
    le = LabelEncoder()

    y = data_df['label'].apply(lambda x: x.split()[0]).values if not dialog else data_df['label']
    y_le = le.fit_transform(y)
    labels = y_le
    class_names = le.classes_

    X = data_df.drop('label', axis=1).values

    if type != 'u':
        if dialog == False:
            unique_classes, class_counts = np.unique(y_le, return_counts=True)
            n_splits = min(2, len(unique_classes))

            skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=False)
            for train_index, test_index in skf.split(X, y_le):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_le[train_index], y_le[test_index]
        else:
            X_train, X_test = X, X
            y_train, y_test = y_le, y_le

        if preprocessing_methods is not None:
            for m in preprocessing_methods:
                X_train, X_test = dataset_preprocessing.apply_preprocessing(m, X_train, X_test, y_train)

    else:
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

            print("Accuracy: ", accuracy)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("F1: ", f1)

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
        for c in classifiers:
            if c == 'kmeans':
                best_num_clusters = unsupervised_models.grid_search_kmeans(X, min_clusters=2, max_clusters=25)
                pred_labels = unsupervised_models.kmeans(X, n_clusters=best_num_clusters)
                print(np.unique(labels))
                print(np.unique(pred_labels))
                unsupervised_models.evaluate_clustering(X, labels, pred_labels)
                unsupervised_models.visualize_clusters_tsne_label(X, pred_labels, y_le, class_names)
            elif c == 'pca':
                """NOT GOOD"""
                X_reduced = unsupervised_models.pca_dimension_reduction(X)
                unsupervised_models.visualize_clusters(c, X_reduced, y_le)

    return 0