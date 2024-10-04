import numpy as np
import pandas as pd
import json
from text_preprocessing import text_preprocessing
from features import save_features

def chapter_features(text, config_path = None):
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

    # Load configurations from JSON file
    with open(config_path, 'r') as f:
        configs = json.load(f)
    feature_specs = configs['chapter_features']

    # Preprocess text as needed
    tokenized_text = text_preprocessing.tokenize_text(text)
    tokenized_text_no_sw = text_preprocessing.tokenize_text(text, remove_stopwords=True)
    tokenized_text_punct = text_preprocessing.tokenize_text(text, include_punctuation=True)
    phrases = text_preprocessing.split_into_phrases(text)

    # Prepare context for feature computation
    context = {
        'text': text,
        'tokenized_text': tokenized_text,
        'tokenized_text_no_sw': tokenized_text_no_sw,
        'tokenized_text_punct': tokenized_text_punct,
        'phrases': phrases,
        'tokens': tokenized_text,
    }

    # Resolve feature specifications
    resolved_feature_specs = {}
    for feature_name, params in feature_specs.items():
        resolved_params = {}
        for param_name, variable_name in params.items():
            resolved_params[param_name] = context.get(variable_name)
        resolved_feature_specs[feature_name] = resolved_params

    # Save features using your existing method
    config = save_features.save_features(feature_specs=resolved_feature_specs)

    # Process config to labels and values, build DataFrame, save to CSV, return config, df
    labels = []
    values = []
    for i in config.items():
        labels.append(i[0])
        values.append(i[1])

    X = np.array(values)
    y = np.array(labels)

    train_df = pd.DataFrame(X)
    train_df['label'] = y
    train_df.to_csv(f'Outputs/chapter_feature.csv', index=False)
    return config, train_df

def sentence_features(text, config_path = None):
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
    # Load configurations from JSON file
    with open(config_path, 'r') as f:
        configs = json.load(f)
    feature_specs = configs['sentence_features']

    phrases = text_preprocessing.split_into_phrases(text)

    # Prepare context for feature computation
    context = {
        'sentences': phrases
    }

    # Resolve feature specifications
    resolved_feature_specs = {}
    for feature_name, params in feature_specs.items():
        resolved_params = {}
        for param_name, variable_name in params.items():
            resolved_params[param_name] = context.get(variable_name)
        resolved_feature_specs[feature_name] = resolved_params

    # Save features using your existing method
    config = save_features.save_features(feature_specs=resolved_feature_specs)

    labels = []
    values = []
    max_length = 0
    for label, value in config.items():
        labels.append(label)
        values.append(value)
        max_length = max(max_length, len(value))

    values_padded = [value + [0] * (max_length - len(value)) for value in values]

    X = np.array(values_padded)
    y = np.array(labels)

    train_df = pd.DataFrame(X)
    train_df['label'] = y
    train_df.to_csv(f'Outputs/sentence_features.csv', index=False)

    return config, train_df

def tf_idf_features(text, config_path = None):
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

    # Load configurations from JSON file
    with open(config_path, 'r') as f:
        configs = json.load(f)
    feature_specs = configs['tf_idf_features']

    # Prepare context for feature computation
    context = {
        'text': text
    }

    # Resolve feature specifications
    resolved_feature_specs = {}
    for feature_name, params in feature_specs.items():
        resolved_params = {}
        for param_name, value in params.items():
            # Directly use the value if it's not a string referencing context
            if isinstance(value, str) and value in context:
                resolved_params[param_name] = context[value]
            else:
                resolved_params[param_name] = value
        resolved_feature_specs[feature_name] = resolved_params

    # Save features using your existing method
    config = save_features.save_features(feature_specs=resolved_feature_specs)

    labels = []
    values = []
    for i in config.items():
        labels.append(i[0])
        values.append(i[1])

    X = np.array(values)
    y = np.array(labels)

    train_df = pd.DataFrame(X)
    train_df['label'] = y
    train_df.to_csv(f'Outputs/tf_idf_features.csv', index=False)
    return config, train_df
