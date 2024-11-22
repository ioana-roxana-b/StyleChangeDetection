import json
import pandas as pd
from classification import classification
from features_methods import extract_all_features
from configs import configs
from features_methods.extract_all_features import create_dfs
from text_preprocessing import text_preprocessing

def dialog():
    """
    Processes dialog-based text features and performs classification using specified classifiers.
    Params:
        None
    Returns:
        None: Outputs the classification results and stores them in CSV files.
    """
    classifier_config_path = 'classification/classifiers_config_dialog.json'

    # Specify the paths for dialog data extraction and output
    text_path = 'Corpus/project-dialogism-novel-corpus-master/data/Emma/quotation_info.csv'
    output_file_path = 'Outputs/Q2/Emma/'

    create_dfs(text_path=text_path, output_file_path=output_file_path, dialogue=True)

    # Path to the combined features file
    # data_path = f'{output_file_path}/all_features.csv'
    # data_df = pd.read_csv(data_path)
    #
    # # Load classifier configurations from the JSON file
    # with open(classifier_config_path, 'r') as f:
    #     classifiers_config = json.load(f)
    #
    # # Iterate over each classifier defined in the configuration
    # for classifier_name, params in classifiers_config.items():
    #     # Extract parameters for the classifier
    #     classification_type = params.get('type', 's')
    #     classifiers_list = params.get('classifiers', [classifier_name])
    #     preprocessing_methods = params.get('preprocessing_methods', [])
    #     dialog = params.get('dialog', True)
    #
    #     # Call the classification function with the extracted parameters
    #     classification.classification(
    #         type=classification_type,
    #         classifiers=classifiers_list,
    #         data_df=data_df,
    #         preprocessing_methods=preprocessing_methods,
    #         dialog=dialog
    #     )


def non_dialog():
    """
       Processes non-dialog-based text features and performs classification using specified classifiers.
       Params:
           None
       Returns:
           None: Outputs classification results and stores them in CSV files.
       """
    # Define the paths for configuration files
    classifier_config_path = 'classification/classifiers_config.json'

    text_path = 'Corpus/Combined_texts/David_Oliver.txt'
    output_file_path = 'Outputs/Q1/David_Oliver/'

    # Split the text into chapters with appropriate labels
    chapters = text_preprocessing.split_into_chapters(text=text_path,label='DOS|TOL|FICTION|NONFICTION'
    )

    create_dfs(text_path=text_path, output_file_path=output_file_path, dialogue=True)

    # Path to the combined features file
    # data_path = f'{output_file_path}/chapter_features.csv'
    # data_df = pd.read_csv(data_path)
    #
    # # Load classifier configurations from JSON file
    # with open(classifier_config_path, 'r') as f:
    #     classifiers_config = json.load(f)
    #
    # # Iterate over each classifier in the configuration
    # for classifier_name, params in classifiers_config.items():
    #     # Extract parameters for the classifier
    #     classification_type = params.get('type', 's')
    #     classifiers_list = params.get('classifiers', [classifier_name])
    #     preprocessing_methods = params.get('preprocessing_methods', [])
    #     dialog = params.get('dialog', False)
    #
    #     # Call the classification function with the extracted parameters
    #     classification.classification(
    #         type=classification_type,
    #         classifiers=classifiers_list,
    #         data_df=data_df,
    #         preprocessing_methods=preprocessing_methods,
    #         dialog=dialog
    #     )

