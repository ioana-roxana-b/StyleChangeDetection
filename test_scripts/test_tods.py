import json
import pandas as pd
from time_series import classification_with_tods


def tods_test():
    """
    Executes time-series classification on a specified dataset using configurations
    defined in an external JSON file. Each classifier's settings, preprocessing steps,
    and feature extraction methods are specified for streamlined testing and evaluation.
    Params:
        None
    Returns:
        None: Outputs classification results and saves them to files or logs them as needed.
    """

    #The path to the data file and classifier configuration JSON file
    data_path = 'Outputs/Dos_Tol/chapter_feature.csv'
    classifier_config_path = 'time_series/tods_classifiers_config.json'

    # Load the dataset into a DataFrame
    data_df = pd.read_csv(data_path)

    # Load classifier configurations from the JSON file
    with open(classifier_config_path, 'r') as f:
        classifiers_config = json.load(f)

    # Iterate over each classifier specified in the configuration file
    for classifier_name, params in classifiers_config.items():
        # Extract configuration details
        classifiers_list = params.get('classifiers', [classifier_name])
        as_time_series = params.get('as_time_series', False)
        preprocessing_methods = params.get('preprocessing_methods', [])
        preprocessing_with_tods = params.get('preprocessing_with_tods', [])

        # Execute the classification function with the extracted parameters
        classification_with_tods.classification_with_tods(
            classifiers=classifiers_list,
            data_df=data_df,
            as_time_series=as_time_series,
            preprocessing_methods=preprocessing_methods,
            preprocessing_with_tods=preprocessing_with_tods
        )
