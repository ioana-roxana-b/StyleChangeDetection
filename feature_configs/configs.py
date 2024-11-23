import numpy as np
import pandas as pd
import json
from text_preprocessing import text_preprocessing
from features_methods import save_features

def chapter_features(text, output_path, chapter_config = 'chapter_features', config_file = 'chapter_configs.json'):
    """
    Computes and saves chapter-level features based on the provided configuration.
    Loads feature configurations, preprocesses the text, computes features, and saves them to a CSV.
    Params:
        text (dict): Dictionary with chapter texts as keys and content as values.
        output_path (str): Directory to save the resulting CSV file.
        chapter_config (str, optional): Key in the JSON config specifying features. Defaults to 'chapter_features'.
        config_file (str, optional): Path to the JSON file with feature settings. Defaults to 'chapter_configs.json'.

    Returns:
        pd.DataFrame: A DataFrame with computed chapter-level features and a 'label' column.
    """
    # Read config file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract feature_specs
    feature_specs = config.get(chapter_config, {})

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
    features = save_features.save_features(feature_specs=resolved_feature_specs)

    # Rest of the code remains the same
    labels = []
    values = []
    for i in features.items():
        labels.append(i[0])
        values.append(i[1])

    X = np.array(values)
    y = np.array(labels)

    train_df = pd.DataFrame(X)
    train_df['label'] = y
    train_df.to_csv(f'{output_path}/{chapter_config}.csv', index=False)

    return train_df


def sentence_features(text, output_path, sentence_config = 'sentence_features', config_file='sentence_configs.json'):
    """
    Computes and saves sentence-level features based on the provided configuration.
    Loads feature configurations, processes sentences from the text, computes features, and saves them to a CSV.
    Params:
        text (dict): Dictionary with chapter texts as keys and content as values.
        output_path (str): Directory to save the resulting CSV file.
        sentence_config (str, optional): Key in the JSON config specifying features. Defaults to 'sentence_features'.
        config_file (str, optional): Path to the JSON file with feature settings. Defaults to 'sentence_configs.json'.

    Returns:
        pd.DataFrame: A DataFrame with computed sentence-level features and a 'label' column.
    """
    # Read config file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract feature_specs
    feature_specs = config.get(sentence_config, {})

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
    features = save_features.save_features(feature_specs=resolved_feature_specs)

    # Rest of the code remains the same
    labels = []
    values = []
    max_length = 0
    for label, value in features.items():
        labels.append(label[0])
        values.append(value)
        max_length = max(max_length, len(value))

    values_padded = [value + [0] * (max_length - len(value)) for value in values]

    X = np.array(values_padded)
    y = np.array(labels)

    train_df = pd.DataFrame(X)
    train_df['label'] = y
    train_df.to_csv(f'{output_path}/{sentence_config}.csv', index=False)

    return train_df

def tf_idf_features(text, output_path, tf_idf_config = 'tf_idf_features', config_file = 'tf_idf_configs.json'):
    """
    Computes and saves TF-IDF features based on the provided configuration.
    Loads feature configurations, computes TF-IDF values for text, and saves the results to a CSV.
    Params:
        text (dict): Chapter texts as a dictionary with chapter names as keys and content as values.
        output_path (str): Directory to save the resulting CSV file.
        tf_idf_config (str, optional): Key in the JSON config specifying features to compute. Defaults to 'all_tf_idf_features'.
        config_file (str, optional): Path to the JSON file with TF-IDF feature settings. Defaults to 'tf_idf_configs.json'.

    Returns:
        pd.DataFrame: A DataFrame with computed TF-IDF features and a 'label' column.
    """
    # Read config file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract feature_specs
    feature_specs = config.get(tf_idf_config, {})

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
    features = save_features.save_features(feature_specs = resolved_feature_specs)

    # Rest of the code remains the same
    labels = []
    values = []
    for i in features.items():
        labels.append(i[0])
        values.append(i[1])

    X = np.array(values)
    y = np.array(labels)

    train_df = pd.DataFrame(X)
    train_df['label'] = y
    train_df.to_csv(f'{output_path}/{tf_idf_config}.csv', index = False)
    return train_df


def chapter_tf_idf(chapter_df, tf_idf_df, output_path):
    """
     Combines chapter-level and TF-IDF features into a single DataFrame.
     Params:
         chapter_path (str): Path to the CSV file containing chapter-level features.
         tf_idf_path (str): Path to the CSV file containing TF-IDF features.
         output_path (str): Path to save the combined CSV file.
     Returns:
         pd.DataFrame: Combined DataFrame containing all features with 'label' as the first column.
     """

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

    combined_df.to_csv(f'{output_path}/chapter_tf_idf.csv', index=False)
    return combined_df

def sentence_chapter(sentence_df, chapter_df, output_path):
    """
    Combines sentence-level and chapter-level features into a single DataFrame.

    Params:
        sentence_df (pd.DataFrame): DataFrame containing sentence-level features.
        chapter_df (pd.DataFrame): DataFrame containing chapter-level features.
        output_path (str): Path to save the combined CSV file.

    Returns:
        pd.DataFrame: Combined DataFrame containing all features with 'label' as the first column.
    """

    # Ensure 'label' is the first column, else reset it to be so
    sentence_df.set_index('label', inplace=True)
    chapter_df.set_index('label', inplace=True)

    # Join the dataframes with suffixes to handle overlapping columns
    combined_df = sentence_df.join(chapter_df, how='left', lsuffix='_sentence', rsuffix='_chapter')

    # Reset the index to move 'label' back to a column
    combined_df.reset_index(inplace=True)

    # Fill missing data with zeros and ensure all data is numeric, except 'label'
    combined_df.fillna(0, inplace=True)
    for col in combined_df.columns:
        if col != 'label':
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)

    # Save the combined DataFrame
    combined_df.to_csv(f'{output_path}/sentence_chapter.csv', index=False)

    return combined_df

def sentence_tf_idf(sentence_df, tf_idf_df, output_path):
    """
    Combines sentence-level and TF-IDF features into a single DataFrame.

    Params:
        sentence_df (pd.DataFrame): DataFrame containing sentence-level features.
        tf_idf_df (pd.DataFrame): DataFrame containing TF-IDF features.
        output_path (str): Path to save the combined CSV file.

    Returns:
        pd.DataFrame: Combined DataFrame containing all features with 'label' as the first column.
    """
    # Ensure 'label' is the first column, else reset it to be so
    sentence_df.set_index('label', inplace=True)
    tf_idf_df.set_index('label', inplace=True)

    # Join the dataframes with suffixes to handle overlapping columns
    combined_df = sentence_df.join(tf_idf_df, how='left', lsuffix='_sentence', rsuffix='_tfidf')

    # Reset the index to move 'label' back to a column
    combined_df.reset_index(inplace=True)

    # Fill missing data with zeros and ensure all data is numeric, except 'label'
    combined_df.fillna(0, inplace=True)
    for col in combined_df.columns:
        if col != 'label':
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)

    # Save the combined DataFrame
    combined_df.to_csv(f'{output_path}/sentence_tf_idf.csv', index=False)

    return combined_df


def all_features(sentence_df, chapter_df, tf_idf_df, output_path):
    """
    Combines sentence-level, chapter-level, and TF-IDF features into a single DataFrame.

    Params:
        sentence_path (str): Path to the CSV file containing sentence-level features.
        chapter_path (str): Path to the CSV file containing chapter-level features.
        tf_idf_path (str): Path to the CSV file containing TF-IDF features.
        output_path (str): Path to save the combined CSV file.

    Returns:
        pd.DataFrame: Combined DataFrame containing all features with 'label' as the first column.
    """
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
    combined_df.to_csv(f'{output_path}/all_features.csv', index=False)

    return combined_df
