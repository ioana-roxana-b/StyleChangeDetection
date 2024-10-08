import json
import pandas as pd
from text_preprocessing import text_preprocessing
from configs import configs
from classification import classification

def dialog():
    """
    Processes dialog-based text features and performs classification using specified classifiers.

    This function extracts dialog features from a specified corpus, applies various feature engineering methods,
    combines the features into a single DataFrame, and then uses a set of classifiers for model training and evaluation.

    Params:
        None

    Returns:
        None: Outputs the classification results and stores them in CSV files.
    """

    # Define the paths for configuration files
    config_path = '../configs/feature_configs.json'
    classifier_config_path = '../classification/classifiers_config_dialog.json'

    # Specify the paths for dialog data extraction and output
    quotation_info_path = '../Corpus/project-dialogism-novel-corpus-master/data/WinnieThePooh/quotation_info.csv'
    output_file_path = '../Outputs/dialogue.txt'

    # Extract and save dialogues from the corpus to an output file
    dialogues = text_preprocessing.extract_and_save_dialogues(quotation_info_path, output_file_path)

    # Extract features using the configurations defined in `feature_configs.json`
    configs.chapter_features(dialogues, config_path)
    configs.sentence_features(dialogues, config_path)
    configs.tf_idf_features(dialogues, config_path)

    # Define the paths to the feature CSV files generated in the previous steps
    sentence_path = '../Outputs/sentence_features.csv'
    chapter_path = '../Outputs/chapter_feature.csv'
    tf_idf_path = '../Outputs/tf_idf_features.csv'

    # Combine all features into a single DataFrame
    configs.all_features(sentence_path, chapter_path, tf_idf_path)

    # Path to the combined features file
    data_path = f'Outputs/all_features_sent.csv'
    data_df = pd.read_csv(data_path)

    # Load classifier configurations from the JSON file
    with open(classifier_config_path, 'r') as f:
        classifiers_config = json.load(f)

    # Iterate over each classifier defined in the configuration
    for classifier_name, params in classifiers_config.items():
        # Extract parameters for the classifier
        classification_type = params.get('type', 's')
        classifiers_list = params.get('classifiers', [classifier_name])
        preprocessing_methods = params.get('preprocessing_methods', [])
        dialog = params.get('dialog', True)

        # Call the classification function with the extracted parameters
        classification.classification(
            type=classification_type,
            classifiers=classifiers_list,
            data_df=data_df,
            preprocessing_methods=preprocessing_methods,
            dialog=dialog
        )


def non_dialog():
    """
       Processes non-dialog-based text features and performs classification using specified classifiers.
       Params:
           None
       Returns:
           None: Outputs classification results and stores them in CSV files.
       """
    # Define the paths for configuration files
    config_path = '../configs/feature_configs.json'
    classifier_config_path = '../classification/classifiers_config.json'

    # Split the text into chapters with appropriate labels
    # chapters = text_preprocessing.split_into_chapters(
    #     text='Corpus/New folder/dickens.txt',
    #     label='DOS|TOL|FICTION|NONFICTION'
    # )
    #
    # # Extract features using the configurations
    # configs.chapter_features(chapters, config_path)
    # configs.sentence_features(chapters, config_path)
    # configs.tf_idf_features(chapters, config_path)
    #
    # # Define the paths to the feature CSV files
    sentence_path = '../Outputs/Fict_Nonfict/sentence_features.csv'
    chapter_path = '../Outputs/Fict_Nonfict/chapter_feature.csv'
    tf_idf_path = '../Outputs/Fict_Nonfict/tf_idf_features.csv'

    # Combine all features into a single DataFrame
    configs.all_features(chapter_path, tf_idf_path)

    # Path to the combined features file
    data_path = '../Outputs/Fict_Nonfict/all_features_combined.csv'
    data_df = pd.read_csv(data_path)

    # Load classifier configurations from JSON file
    with open(classifier_config_path, 'r') as f:
        classifiers_config = json.load(f)

    # Iterate over each classifier in the configuration
    for classifier_name, params in classifiers_config.items():
        # Extract parameters for the classifier
        classification_type = params.get('type', 's')
        classifiers_list = params.get('classifiers', [classifier_name])
        preprocessing_methods = params.get('preprocessing_methods', [])
        dialog = params.get('dialog', False)

        # Call the classification function with the extracted parameters
        classification.classification(
            type=classification_type,
            classifiers=classifiers_list,
            data_df=data_df,
            preprocessing_methods=preprocessing_methods,
            dialog=dialog
        )


