import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

from models import supervised_models, unsupervised_models, visualization
from features_methods import feature_engineering

THRESHOLD = 20

def classification(type, classifiers, data_df, preprocessing_methods = None, dialog = False, parameters=None, viz = 'tsne'):
    """
      Performs classification using specified supervised or unsupervised models on the given data.
      It preprocesses the data, splits it into training and test_scripts sets, applies the specified models, and evaluates their
      performance using metrics such as accuracy, precision, recall, and F1 score for supervised learning, or clustering
      metrics for unsupervised models. Optionally, it also visualizes the results.

      Params:
          type (str): Specifies whether to perform 's' (supervised) or 'u' (unsupervised) classification.
          classifiers (list): List of classifier names to be applied (e.g., ['random_forest', 'svm'] for supervised or
                              ['kmeans', 'pca'] for unsupervised).
          data_df (pd.DataFrame): Input data in the form of a DataFrame, where the last column ('label') is used as target labels.
          preprocessing_methods (list, optional): List of preprocessing methods to apply (e.g., ['pca', 'minmax_sc']).
          dialog (bool): Flag to indicate if the labels should be filtered based on a minimum frequency (used for dialogism corpus).
          parameters (dict, optional): Parameters for the classifiers.
      Returns:
          int: Always returns 0 after execution.

      Workflow:
          1. If `dialog` is True, filter labels with at least 20 occurrences (keep only characters with at least 20 lines of dialogue).
          2. Encode labels using `LabelEncoder`.
          3. Perform data preprocessing and train-test_scripts splitting based on the `type` parameter:
             - For supervised models ('s'), use `StratifiedKFold` or `ShuffleSplit`.
             - For unsupervised models ('u'), use `train_test_split`.
          4. Apply specified preprocessing methods.
          5. Run the specified models and evaluate:
             - For supervised models: Compute and print accuracy, precision, recall, and F1 score.
             - For unsupervised models: Perform clustering and visualize results using t-SNE or PCA.
      """

    if dialog:
        label_counts = data_df['label'].value_counts()
        labels_to_keep = label_counts[label_counts >= THRESHOLD].index
        filtered_data_df = data_df[data_df['label'].isin(labels_to_keep)]
        if filtered_data_df.empty:
            raise ValueError("No data available after filtering. Adjust the threshold or check the data.")

        le = LabelEncoder()
        y = filtered_data_df['label']
        y_le = le.fit_transform(y)
        labels = y_le
        class_names = le.classes_
        X = filtered_data_df.drop('label', axis=1).values

    else:
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
            shuffle_split = ShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
            for train_index, test_index in shuffle_split.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_le[train_index], y_le[test_index]

        if preprocessing_methods is not None:
            for preprocessing in preprocessing_methods:
                method = preprocessing["method"]
                params = preprocessing["parameters"]
                X_train, X_test = feature_engineering.apply_preprocessing(method, X_train, X_test, y_train, **params)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y_le, test_size=0.4, random_state=42)
        if preprocessing_methods is not None:
            for preprocessing in preprocessing_methods:
                method = preprocessing["method"]
                params = preprocessing["parameters"]
                X, _ = feature_engineering.apply_preprocessing(method, X, y_train=y_le, **params)

    if type == 's':
        for c in classifiers:
            params = parameters.get(c, {})  # Extract parameters for this classifier
            clf, y_pred = supervised_models.sup_models(X_train, y_train, X_test, c=c, **params)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
            f1 = f1_score(y_test, y_pred, average='macro')

           # print("Real values: ", y_test)
           # print("Pred values: ", y_pred)

            print(f"\nClassifier: {c}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            results_df = pd.DataFrame({
                'Classifier': [c],
                'Preprocessing methods': [preprocessing_methods],
                'Accuracy': [accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'F1 Score': [f1]
            })
            results_df.to_csv(f'Outputs/Results/results_{c}.csv', mode='a', index=False)
    elif type == 'u':
        for c in classifiers:
            if c == 'kmeans':
                best_num_clusters = unsupervised_models.grid_search_kmeans(X, min_clusters=2, max_clusters=25)
                pred_labels = unsupervised_models.kmeans(X, n_clusters=best_num_clusters)
                print(np.unique(labels))
                print(np.unique(pred_labels))
                unsupervised_models.evaluate_clustering(X, labels, pred_labels)
                visualization.visualize_clusters(X, pred_labels, y_le, class_names, viz)
            elif c == 'pca':
                """NOT GOOD"""
                X_reduced = unsupervised_models.pca_dimension_reduction(X)
                visualization.visualize_clusters(c, X_reduced, y_le)

    return 0