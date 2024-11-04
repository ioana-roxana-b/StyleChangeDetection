import json
import pandas as pd
from time_series import classification_with_tods


def tods_test():
    data_path = 'Outputs/Dos_Tol/chapter_feature.csv'  # Path to your data file
    classifier_config_path = 'time_series/tods_classifiers_config.json'  # Path to your configuration file

    # Load the dataset
    data_df = pd.read_csv(data_path)

    # Load TODS classifier configurations from a JSON file
    with open(classifier_config_path, 'r') as f:
        classifiers_config = json.load(f)

    # Iterate over each classifier defined in the configuration
    for classifier_name, params in classifiers_config.items():
        # Extract parameters for the classifier
        classifiers_list = params.get('classifiers', [classifier_name])
        as_time_series = params.get('as_time_series', False)
        preprocessing_methods = params.get('preprocessing_methods', [])
        preprocessing_with_tods = params.get('preprocessing_with_tods', [])

        # Call the classification_with_tods function with the extracted parameters
        classification_with_tods.classification_with_tods(
            classifiers=classifiers_list,
            data_df=data_df,
            as_time_series=as_time_series,
            preprocessing_methods=preprocessing_methods,
            preprocessing_with_tods=preprocessing_with_tods
        )