import numpy as np
import pandas as pd
import json
from text_preprocessing import text_preprocessing
from features_methods import save_features

def chapter_features(text, config_file = 'feature_config.json'):
    """
    Computes and saves chapter-level features based on the provided configuration.
    Loads feature configurations, preprocesses the text, computes features, and saves them to a CSV.
    Params:
        text (dict): Dictionary containing chapter texts.
        config_path (str): Path to the JSON configuration file specifying the features.
    Returns:
        tuple:
            - config (dict): Dictionary of computed chapter-level features.
            - train_df (pd.DataFrame): DataFrame containing the features and labels.
    """
    # Read config file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract feature_specs
    feature_specs = config.get('chapter_features', {})

    # Prepare variables needed
    variables = {}
    variables['text'] = text
    variables['tokenized_text'] = text_preprocessing.tokenize_text(text)
    variables['tokenized_text_no_sw'] = text_preprocessing.tokenize_text(text, remove_stopwords=True)
    variables['tokenized_text_punct'] = text_preprocessing.tokenize_text(text, include_punctuation=True)
    variables['phrases'] = text_preprocessing.split_into_phrases(text)

    # Resolve variables in params
    resolved_feature_specs = {}
    for feature_name, feature_info in feature_specs.items():
        function_name = feature_info.get('function')
        params = feature_info.get('params', {})
        resolved_params = {}
        for param_name, var_name in params.items():
            if var_name in variables:
                resolved_params[param_name] = variables[var_name]
            else:
                raise ValueError(f"Variable {var_name} not found in variables.")
        resolved_feature_specs[feature_name] = {
            'function': function_name,
            'params': resolved_params
        }

    # Call save_features
    config = save_features.save_features(feature_specs=resolved_feature_specs)

    # Rest of the code remains the same
    labels = []
    values = []
    for i in config.items():
        labels.append(i[0])
        values.append(i[1])

    X = np.array(values)
    y = np.array(labels)

    train_df = pd.DataFrame(X)
    train_df['label'] = y
    train_df.to_csv('Outputs/Fict_Nonfict/chapter_feature.csv', index=False)
    return config, train_df


def sentence_features(text, config_file='feature_config.json'):
    """
    Computes and saves sentence-level features based on the provided configuration.
    Loads feature configurations, processes sentences from the text, computes features, and saves them to a CSV.
    Params:
        text (dict): Dictionary containing chapter texts.
        config_path (str): Path to the JSON configuration file specifying the features.
    Returns:
        tuple:
            - config (dict): Dictionary of computed sentence-level features.
            - train_df (pd.DataFrame): DataFrame containing the features and labels.
    """
    # Read config file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract feature_specs
    feature_specs = config.get('sentence_features', {})

    # Prepare variables needed
    variables = {}
    variables['text'] = text
    variables['phrases'] = text_preprocessing.split_into_phrases(text)
    variables['sentences'] = variables['phrases']

    # Resolve variables in params
    resolved_feature_specs = {}
    for feature_name, feature_info in feature_specs.items():
        function_name = feature_info.get('function')
        params = feature_info.get('params', {})
        resolved_params = {}
        for param_name, var_name in params.items():
            if var_name in variables:
                resolved_params[param_name] = variables[var_name]
            else:
                raise ValueError(f"Variable {var_name} not found in variables.")
        resolved_feature_specs[feature_name] = {
            'function': function_name,
            'params': resolved_params
        }

    # Call save_features
    config = save_features.save_features(feature_specs=resolved_feature_specs)

    # Rest of the code remains the same
    labels = []
    values = []
    max_length = 0
    for label, value in config.items():
        labels.append(label[0])
        values.append(value)
        max_length = max(max_length, len(value))

    values_padded = [value + [0] * (max_length - len(value)) for value in values]

    X = np.array(values_padded)
    y = np.array(labels)

    train_df = pd.DataFrame(X)
    train_df['label'] = y
    train_df.to_csv('Outputs/Fict_Nonfict/sentence_features.csv', index=False)

    return config, train_df

def tf_idf_features(text, config_file='feature_config.json'):
    """
    Computes and saves TF-IDF features based on the provided configuration.
    Loads feature configurations, computes TF-IDF values for text, and saves the results to a CSV.
    Params:
        text (dict): Dictionary containing chapter texts.
        config_path (str): Path to the JSON configuration file specifying the features.
    Returns:
        tuple:
            - config (dict): Dictionary of computed TF-IDF features.
            - train_df (pd.DataFrame): DataFrame containing the features and labels.
    """
    # Read config file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract feature_specs
    feature_specs = config.get('tf_idf_features', {})

    # Prepare variables
    variables = {}
    variables['text'] = text

    # Resolve variables in params
    resolved_feature_specs = {}
    for feature_name, feature_info in feature_specs.items():
        function_name = feature_info.get('function')
        params = feature_info.get('params', {})
        resolved_params = {}
        for param_name, var_value in params.items():
            if isinstance(var_value, str) and var_value in variables:
                resolved_params[param_name] = variables[var_value]
            else:
                resolved_params[param_name] = var_value
        resolved_feature_specs[feature_name] = {
            'function': function_name,
            'params': resolved_params
        }

    # Call save_features
    config = save_features.save_features(feature_specs = resolved_feature_specs)

    # Rest of the code remains the same
    labels = []
    values = []
    for i in config.items():
        labels.append(i[0])
        values.append(i[1])

    X = np.array(values)
    y = np.array(labels)

    train_df = pd.DataFrame(X)
    train_df['label'] = y
    train_df.to_csv('Outputs/Fict_Nonfict/tf_idf_features.csv', index = False)
    return config, train_df


def all_features(chapter_path, tf_idf_path):
    """
     Combines chapter-level and TF-IDF features into a single DataFrame.
     Params:
         chapter_path (str): Path to the CSV file containing chapter-level features.
         tf_idf_path (str): Path to the CSV file containing TF-IDF features.

     Returns:
         pd.DataFrame: Combined DataFrame containing all features with 'label' as the first column.
     """
    chapter_df = pd.read_csv(chapter_path)
    tf_idf_df = pd.read_csv(tf_idf_path)

    chapter_df.set_index('label', inplace=True)
    tf_idf_df.set_index('label', inplace=True)

    # Combine the dataframes, adding suffixes to overlapping columns
    combined_df = chapter_df.join(tf_idf_df, how='left', lsuffix='_chapter', rsuffix='_tfidf')

    # Reset the index to move 'label' back to a column
    combined_df.reset_index(inplace=True)

    combined_df.fillna(0, inplace=True)
    for col in combined_df.columns:
        if col != 'label':
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)

    combined_df.to_csv('Outputs/Fict_Nonfict/all_features.csv', index=False)
    return combined_df

def all_features_v2(sentence_path, chapter_path, tf_idf_path):
    """
    Combines sentence-level, chapter-level, and TF-IDF features into a single DataFrame.

    Params:
        sentence_path (str): Path to the CSV file containing sentence-level features.
        chapter_path (str): Path to the CSV file containing chapter-level features.
        tf_idf_path (str): Path to the CSV file containing TF-IDF features.

    Returns:
        pd.DataFrame: Combined DataFrame containing all features with 'label' as the first column.
    """

    # Load data from CSV files
    sentence_df = pd.read_csv(sentence_path)
    chapter_df = pd.read_csv(chapter_path)
    tf_idf_df = pd.read_csv(tf_idf_path)

    # Ensure 'label' is the first column, else reset it to be so
    sentence_df.set_index('label', inplace=True)
    chapter_df.set_index('label', inplace=True)
    tf_idf_df.set_index('label', inplace=True)

    # Join the dataframes
    combined_df = sentence_df.join([chapter_df, tf_idf_df], how='left')

    # Reset the index to move 'label' back to a column
    combined_df.reset_index(inplace=True)

    # Fill missing data with zeros and ensure all data is numeric, except 'label'
    combined_df.fillna(0, inplace=True)
    for col in combined_df.columns:
        if col != 'label':
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)

    # Save the combined DataFrame
    combined_df.to_csv('Outputs/Fict_Nonfict/all_features_combined.csv', index=False)

    return combined_df
