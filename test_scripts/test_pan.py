import json
import os

import pandas as pd
from classification import classification_pan
from pan import pipeline_pan

def process_classifier_config(config_key, config, train_features, test_features):
    """
    Processes a single classifier configuration specified by a key from the JSON structure and
    performs classification using the defined parameters.

    Params:
        config_key (str): The key corresponding to the specific classifier configuration to process.
        config (dict): A dictionary containing the full configuration loaded from the JSON file.
        data_df (pd.DataFrame): The dataset to be used for classification.
    """
    # Get the specific configuration dictionary
    classifier_config = config.get(config_key, {})

    # Ensure the selected configuration is a dictionary
    if not isinstance(classifier_config, dict):
        raise ValueError(f"The key '{config_key}' does not point to a valid dictionary in the configuration file.")

    # Iterate over each classifier within the selected configuration
    for classifier_name, classifier_details in classifier_config.items():
        if isinstance(classifier_details, dict) and 'type' in classifier_details:
            # Extract basic parameters for the classifier
            classifiers_list = classifier_details.get('classifiers', [classifier_name])
            preprocessing_methods_config = classifier_details.get('preprocessing_methods', [])
            classifier_parameters = classifier_details.get('parameters', {})

            # Parse preprocessing methods and their parameters
            preprocessing_methods = [
                {
                    "method": method_config["method"],
                    "parameters": method_config.get("parameters", {})
                }
                for method_config in preprocessing_methods_config
            ]

            # Call the classification function with the extracted parameters
            classification_pan.classification(
                classifiers=classifiers_list,
                train_features=train_features,
                test_features = test_features,
                preprocessing_methods=preprocessing_methods,
                parameters=classifier_parameters
            )
        else:
            raise ValueError(f"Invalid configuration format for classifier '{classifier_name}' in '{config_key}'.")

def load_feature_files(train_dir, test_dir):
    """
    Load feature CSV files from both train and test directories and save the concatenated files.
    """

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory '{train_dir}' does not exist.")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory '{test_dir}' does not exist.")

    train_feature_files = []
    test_feature_files = []

    for f in os.listdir(train_dir):
        if f.endswith('.csv'):
            train_feature_files.append(os.path.join(train_dir, f))

    for f in os.listdir(test_dir):
        if f.endswith('.csv'):
            test_feature_files.append(os.path.join(test_dir, f))

    if not train_feature_files:
        raise FileNotFoundError(f"No CSV files found in train directory '{train_dir}'.")
    if not test_feature_files:
        raise FileNotFoundError(f"No CSV files found in test directory '{test_dir}'.")

    # Load and concatenate the dataframes
    train_features = pd.concat([pd.read_csv(f) for f in train_feature_files], ignore_index=False)
    test_features = pd.concat([pd.read_csv(f) for f in test_feature_files], ignore_index=False)

    #print(train_features.shape, test_features.shape)

    # FOD DEBUGGING
    # train_output_path = os.path.join(train_dir, "concatenated_train_features.csv")
    # test_output_path = os.path.join(test_dir, "concatenated_test_features.csv")

    # train_features.to_csv(train_output_path, index=False)
    # test_features.to_csv(test_output_path, index=False)
    #
    # print(f"Concatenated train features saved to: {train_output_path}")
    # print(f"Concatenated test features saved to: {test_output_path}")

    return train_features, test_features


def test_pan(train_dataset_path,test_dataset_path, train_truth_path, test_truth_path, generate_features,
            features_path_train, features_path_test, classifier_config_path, classifier_config_key, language):
    """
    Process classifier based on configuration and perform classification.
    """
    if generate_features:
        pipeline_pan.pipeline_pan(train_dataset_path, test_dataset_path, train_truth_path, 
                                  test_truth_path, features_path_train, features_path_test, language)

    # Load the classifier configuration
    with open(classifier_config_path, 'r') as f:
        classifiers_config = json.load(f)

    # Load feature files from train and test directories
    train_features, test_features = load_feature_files(features_path_train, features_path_test)

    # Process the classifier configuration and perform classification
    process_classifier_config(classifier_config_key, classifiers_config, train_features, test_features)

