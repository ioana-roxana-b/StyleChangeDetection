import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from features_methods import feature_engineering
from tods.sk_interface.detection_algorithm.IsolationForest_skinterface import IsolationForestSKI
from tods.sk_interface.detection_algorithm.LSTMODetector_skinterface import LSTMODetectorSKI

THRESHOLD = 20
classifier_mapping = {
    'IsolationForestSKI': IsolationForestSKI,
    'LSTMODetectorSKI': LSTMODetectorSKI,
    # Add other classifiers here
}
def classification_with_tods(classifiers, data_df, dialog=False, as_time_series=False,
                             preprocessing_methods=None, preprocessing_with_tods=None):
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

    unique_classes, class_counts = np.unique(y_le, return_counts=True)
    n_splits = min(2, len(unique_classes))

    # Data splitting using StratifiedKFold or ShuffleSplit
    skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=False)
    for train_index, test_index in skf.split(X, y_le):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_le[train_index], y_le[test_index]

    # Apply custom preprocessing methods if provided
    if preprocessing_methods is not None:
        for method in preprocessing_methods:
            X_train, X_test = feature_engineering.apply_preprocessing(method, X_train, X_test, y_train)

    # Convert data to time-series format if as_time_series is True
    if as_time_series:
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

    # Apply TODS classifiers
    for classifier_name in classifiers:
        try:
            # Retrieve the classifier class from the mapping
            classifier_class = classifier_mapping.get(classifier_name)
            if classifier_class is None:
                print(f"Classifier '{classifier_name}' is not in the classifier mapping.")
                continue

            # Instantiate the classifier
            transformer = classifier_class()

            # Fit and predict
            transformer.fit(X_train)
            y_pred = transformer.predict(X_test)
            y_score = transformer.predict_score(X_test)

            # Print evaluation metrics
            print(f"Classifier: {classifier_name}")
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

            # ROC Curve
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.title(f'ROC Curve for {classifier_name}')
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            plt.legend(loc='lower right')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.show()

        except Exception as e:
            print(f"An error occurred while using the classifier '{classifier_name}': {e}")

    return 0