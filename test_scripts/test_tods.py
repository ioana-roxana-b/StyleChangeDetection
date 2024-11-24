import json
import pandas as pd

from feature_configs import configs
from features_methods.extract_all_features import create_dfs
from text_preprocessing import text_preprocessing
from time_series import classification_with_tods

def tods_test(text = None):
    """
    Executes time-series classification on a specified dataset using configurations
    defined in an external JSON file. Each classifier's settings, preprocessing steps,
    and feature extraction methods are specified for streamlined testing and evaluation.
    Params:
        None
    Returns:
        None: Outputs classification results and saves them to files or logs them as needed.
    """

    # The path to the data file and classifier configuration JSON file
    config_path = 'feature_configs/chapter_configs.json'
    classifier_config_path = f'time_series/tods_classifiers_config.json'

    text_path = f'Corpus/Combined_texts/{text}.txt'
    output_file_path = f'Outputs/Q1/{text}/'

    # create_dfs(text_path=text_path, output_file_path=output_file_path)

    # Path to the combined features file
    data_path = f'{output_file_path}/chapter_features.csv'
    data_df = pd.read_csv(data_path)


    # Load classifier configurations from the JSON file
    with open(classifier_config_path, 'r') as f:
        classifiers_config = json.load(f)

        # Iterate over each classifier specified in the configuration file
    for classifier_name, params in classifiers_config.items():
        try:
            # Extract configuration details
            classifiers_list = params.get('classifiers', [classifier_name])
            as_time_series = params.get('as_time_series', False)
            preprocessing_methods_config = params.get('preprocessing_methods', [])
            preprocessing_methods = [
                {
                    "method": method_config["method"],
                    "parameters": method_config.get("parameters", {})
                }
                for method_config in preprocessing_methods_config
            ]
            processing_methods_from_tods = params.get('preprocessing_with_tods', [])
            time_series_preprocessing = params.get('time_series_preprocessing', [])

            # Execute the classification function with the extracted parameters
            classification_with_tods.classification_with_tods(
                classifiers=classifiers_list,
                data_df=data_df,
                as_time_series=as_time_series,
                preprocessing_methods=preprocessing_methods,
                processing_methods_from_tods=processing_methods_from_tods,
                time_series_preprocessing=time_series_preprocessing,
                parameters=params.get("parameters", {})
            )
        except Exception as e:
            print(f"Error in classification for {classifier_name}: {e}")