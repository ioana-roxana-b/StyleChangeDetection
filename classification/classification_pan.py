import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from features_methods import feature_engineering
from models import supervised_models_pan
import re

def compute_c_at_1(y_true, y_pred_proba):
    """
    Compute the c@1 metric as described in the problem.
    """
    n = len(y_true)
    # Correctly classified problems
    nc = 0

    # Unanswered problems
    nu = 0

    for true, prob in zip(y_true, y_pred_proba):
        if prob == 0.5:
            nu += 1
        elif (prob > 0.5 and true == 1) or (prob < 0.5 and true == 0):
            nc += 1

    c_at_1 = (1 / n) * (nc + (nu * nc / n))
    return c_at_1


def extract_problem_id_from_label(label):
    """
    Extracts the problem ID (e.g., 'EE002') from the label column.
    """
    parts = label.split('_')
    if len(parts) > 1:
        return parts[1]
    return None

def classification(classifiers, train_features, test_features, preprocessing_methods=None, parameters=None, output_file="answers.txt"):
    """
    Performs classification and outputs results in answers.txt, along with evaluation metrics.
    """
    os.makedirs("Outputs/Results/PAN/", exist_ok=True)

    # Preprocess labels
    train_features['binary_label'] = train_features['label'].str[0].map({'Y': 1, 'N': 0})
    test_features['binary_label'] = test_features['label'].str[0].map({'Y': 1, 'N': 0})

    # Extract or validate 'problem_id' column
    if 'problem_id' not in test_features.columns:
        test_features['problem_id'] = test_features['label'].apply(extract_problem_id_from_label)

    if test_features['problem_id'].isnull().any():
        test_features['problem_id'] = [f"ID_{i}" for i in range(len(test_features))]

    y_train = train_features['binary_label'].values
    y_test = test_features['binary_label'].values

    X_train = train_features.drop(columns=['label', 'binary_label'], errors='ignore')
    X_test = test_features.drop(columns=['label', 'binary_label'], errors='ignore')

    # Ensure only numeric columns are passed to the model
    X_train = X_train.select_dtypes(include=[np.number]).values
    X_test = X_test.select_dtypes(include=[np.number]).values

    # Handle missing or infinite values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply preprocessing
    if preprocessing_methods:
        for preprocessing in preprocessing_methods:
            method = preprocessing["method"]
            params = preprocessing.get("parameters", {})
            X_train, X_test = feature_engineering.apply_preprocessing(method, X_train, X_test, y_train, **params)

    # Train classifiers and evaluate
    prob_scores = []
    metrics = []
    for c in classifiers:
        # Train and predict
        clf, y_pred_proba = supervised_models_pan.sup_models(X_train, y_train, X_test, c=c, **parameters)

        # Convert probabilities to binary predictions for metrics
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_pred_proba)  # Use probabilities
        c_at_1 = compute_c_at_1(y_test, y_pred_proba)
        pan_score = auc*c_at_1

        # Log metrics
        metrics.append({
            'Classifier': c,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc,
            'c@1': c_at_1,
            'PAN score': pan_score
        })

        print(f"\nClassifier: {c}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"c@1: {c_at_1:.4f}")
        print(f"PAN score: {pan_score:.4f}")

        # Append probabilities
        prob_scores.append((c, y_pred_proba))

    # Write answers.txt
    with open(output_file, 'w') as f:
        for problem_id, prob in zip(test_features['problem_id'], prob_scores[0][1]):
            f.write(f"{problem_id} {prob:.3f}\n")

    return 0
