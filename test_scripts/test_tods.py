import json
import pandas as pd

from configs import configs
from text_preprocessing import text_preprocessing
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

    # The path to the data file and classifier configuration JSON file
    config_path = 'configs/chapter_configs.json'
    classifier_config_path = 'time_series/tods_classifiers_config.json'


    # # Split the text into chapters with appropriate labels
    # chapters = text_preprocessing.split_into_chapters(
    #     text='Corpus/New folder/Dos_Tol.txt',
    #     label='DOS|TOL|FICTION|NONFICTION|Shakespeare|Fletcher'
    # )
    #
    # # Extract features using the configurations
    # configs.chapter_features(chapters, config_path)
    # configs.sentence_features(chapters, config_path)
    # configs.tf_idf_features(chapters, config_path)
    # #
    # # Define the paths to the feature CSV files
    # sentence_path = 'Outputs/TNK/sentence_features.csv'
    # chapter_path = 'Outputs/Dos_Tol/chapter_feature.csv'
    # tf_idf_path = 'Outputs/Dos_Tol/tf_idf_features.csv'
    #
    # #Combine all features into a single DataFrame
    # configs.all_features_v2(sentence_path,chapter_path, tf_idf_path)

    # Path to the combined features file
    data_path = 'Outputs/Dos_Tol/all_features_combined.csv'

    # Load the dataset into a DataFrame
    data_df = pd.read_csv(data_path, dtype={'column_name': 'float32'}, low_memory=True)


    # Load classifier configurations from the JSON file
    with open(classifier_config_path, 'r') as f:
        classifiers_config = json.load(f)

    # Iterate over each classifier specified in the configuration file
    for classifier_name, params in classifiers_config.items():
        # Extract configuration details
        classifiers_list = params.get('classifiers', [classifier_name])
        as_time_series = params.get('as_time_series', False)
        preprocessing_methods = params.get('preprocessing_methods', [])
        processing_methods_from_tods = params.get('preprocessing_with_tods', [])
        time_series_preprocessing = params.get('time_series_preprocessing', [])

        # Execute the classification function with the extracted parameters
        classification_with_tods.classification_with_tods(
            classifiers=classifiers_list,
            data_df=data_df,
            as_time_series=as_time_series,
            preprocessing_methods=preprocessing_methods,
            processing_methods_from_tods=processing_methods_from_tods,
            time_series_preprocessing = time_series_preprocessing
        )
