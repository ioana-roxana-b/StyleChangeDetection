import json
import os

import pandas as pd
from classification import classification
from features_methods.extract_all_features import create_dfs
from graph import pipeline_wan


def process_classifier_config(config_key, config, data_df, viz = 'tsne'):
    """
    Processes a single classifier configuration specified by a key from the JSON structure and
    performs classification using the defined parameters.

    Params:
        config_key (str): The key corresponding to the specific classifier configuration to process.
        config (dict): A dictionary containing the full configuration loaded from the JSON file.
        data_df (pd.DataFrame): The dataset to be used for classification.

    Returns:
        None: Invokes the classification function for the specified configuration and outputs the
              results. Results are typically stored or processed as defined in the classification logic.
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
            classification_type = classifier_details.get('type', 's')
            classifiers_list = classifier_details.get('classifiers', [classifier_name])
            preprocessing_methods_config = classifier_details.get('preprocessing_methods', [])
            classifier_parameters = classifier_details.get('parameters', {})
            dialog = classifier_details.get('dialog', True)

            # Parse preprocessing methods and their parameters
            preprocessing_methods = [
                {
                    "method": method_config["method"],
                    "parameters": method_config.get("parameters", {})
                }
                for method_config in preprocessing_methods_config
            ]

            # Call the classification function with the extracted parameters
            classification.classification(
                type=classification_type,
                classifiers=classifiers_list,
                data_df=data_df,
                preprocessing_methods=preprocessing_methods,
                dialog=dialog,
                parameters=classifier_parameters,
                viz = viz
            )
        else:
            raise ValueError(f"Invalid configuration format for classifier '{classifier_name}' in '{config_key}'.")

def test(problem, text_name, input_text_path, generate_features, features_path,
         classifier_config_path, classifier_config_key, label, language):
    """
    Process classifier based on configuration and perform classification.
    """
    directory = os.path.dirname(features_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if generate_features and problem == 'normal':
        create_dfs(text_path=input_text_path, output_file_path=directory)
    elif generate_features and problem == 'dialogism':
        create_dfs(text_path=input_text_path, output_file_path=directory, dialogue=True)
    elif generate_features and problem == 'wan':
        pipeline_wan.pipeline_wan(text_name, input_text_path, label, language)
    data_df = pd.read_csv(features_path)

    # Load classifier configurations from JSON file
    with open(classifier_config_path, 'r') as f:
        classifiers_config = json.load(f)

    # Call the function to process the loaded JSON configuration
    process_classifier_config(classifier_config_key, classifiers_config, data_df)