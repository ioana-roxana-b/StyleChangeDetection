import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
from features_methods import feature_engineering

from tods.sk_interface.detection_algorithm.ABOD_skinterface import ABODSKI
from tods.sk_interface.detection_algorithm.AutoEncoder_skinterface import AutoEncoderSKI
from tods.sk_interface.detection_algorithm.IsolationForest_skinterface import IsolationForestSKI
from tods.sk_interface.detection_algorithm.LSTMODetector_skinterface import LSTMODetectorSKI

from tods.sk_interface.data_processing.TimeIntervalTransform_skinterface import TimeIntervalTransformSKI
from tods.sk_interface.data_processing.TimeStampValidation_skinterface import TimeStampValidationSKI

from tods.sk_interface.timeseries_processing.SKAxiswiseScaler_skinterface import SKAxiswiseScalerSKI
from tods.sk_interface.timeseries_processing.SKPowerTransformer_skinterface import SKPowerTransformerSKI

# Constants and mappings
classifier_mapping = {
    'ABODSKI': ABODSKI,
    'AutoEncoderSKI': AutoEncoderSKI,
    'IsolationForestSKI': IsolationForestSKI,
    'LSTMODetectorSKI': LSTMODetectorSKI,
}

preprocessing_mapping_tods = {
    'TimeIntervalTransformSKI': TimeIntervalTransformSKI,
    'TimeStampValidationSKI': TimeStampValidationSKI,
}

time_series_processing_mapping = {
    'SKAxiswiseScalerSKI': SKAxiswiseScalerSKI,
    'SKPowerTransformerSKI': SKPowerTransformerSKI,
}

def classification_with_tods(classifiers, data_df, as_time_series=False,
                             preprocessing_methods=None, processing_methods_from_tods=None, time_series_preprocessing=None, parameters=None):
    """
    Executes anomaly detection or time-series classification using specified TODS classifiers on a dataset.
    Applies optional preprocessing methods and handles data splitting into training and testing sets.

    Params:
        classifiers (list): List of classifier names to apply (e.g., ['IsolationForestSKI']).
        data_df (pd.DataFrame): Input data with 'label' as the target column.
        as_time_series (bool): If True, reshapes data to 3D format for time-series analysis.
        preprocessing_methods (list, optional): Custom preprocessing methods.
        processing_methods_from_tods (list, optional): TODS preprocessing methods.
        time_series_preprocessing (list, optional): TODS time-series-specific preprocessing methods.
        parameters (dict, optional): Additional parameters for classifiers.

    Returns:
        int: Returns 0 on successful execution.
    """
    le = LabelEncoder()
    y = data_df['label'].apply(lambda x: x.split()[0]).values
    y_le = le.fit_transform(y)
    class_names = le.classes_
    X = data_df.drop('label', axis=1).values

    unique_classes, class_counts = np.unique(y_le, return_counts=True)
    n_splits = min(2, len(unique_classes))


    X_train, X_test, y_train, y_test = train_test_split(X, y_le, test_size=0.4, random_state=42)

    # Apply custom preprocessing methods
    if preprocessing_methods:
        for preprocessing in preprocessing_methods:
            method = preprocessing["method"]
            params = preprocessing["parameters"]
            X_train, X_test = feature_engineering.apply_preprocessing(method, X_train, X_test, y_train, **params)

    # Time-series preprocessing
    if as_time_series and time_series_preprocessing:
        for method_name in time_series_preprocessing:
            method_class = time_series_processing_mapping.get(method_name)
            if method_class:
                print(f"Applying time-series preprocessing: {method_name}")
                method_instance = method_class()
                X_train = method_instance.produce(X_train)
                X_test = method_instance.produce(X_test)

    # TODS-specific preprocessing
    if processing_methods_from_tods:
        for method_name in processing_methods_from_tods:
            method_class = preprocessing_mapping_tods.get(method_name)
            if method_class:
                print(f"Applying TODS preprocessing method: {method_name}")
                method_instance = method_class()
                X_train = method_instance.produce(X_train).value
                X_test = method_instance.produce(X_test).value

    # Apply classifiers
    for classifier_name in classifiers:
        try:
            # Get classifier class
            classifier_class = classifier_mapping.get(classifier_name)
            if not classifier_class:
                print(f"Classifier '{classifier_name}' not found in mapping.")
                continue

            # Instantiate and fit classifier
            transformer = classifier_class(**(parameters or {}))
            transformer.fit(X_train)
            y_pred = transformer.predict(X_test)
            y_score = getattr(transformer, 'predict_score', lambda x: None)(X_test)

            # Evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            print(f"\nClassifier: {classifier_name}")
            print("Accuracy:", accuracy)
            print("Classification Report:\n", classification_report(y_test, y_pred))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

            # Save results
            results_df = pd.DataFrame({
                'Classifier': [classifier_name],
                'Preprocessing methods': [preprocessing_methods],
                'Accuracy': [accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'F1 Score': [f1]
            })
            os.makedirs('Outputs/Results', exist_ok=True)
            results_df.to_csv(f'Outputs/Results/results_{classifier_name}.csv', mode='a', index=False)

            # Plot ROC curve (if scores available)
            if y_score is not None:
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                plt.figure()
                plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                plt.title(f'ROC Curve for {classifier_name}')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc='lower right')
                plt.show()

        except Exception as e:
            print(f"Error with classifier '{classifier_name}': {e}")

    return 0
