import numpy as np
import pandas as pd
import dataset
import text_preprocessing

def chapter_features(text):
    tokenized_text = text_preprocessing.tokenize_text(text)
    tokenized_text_no_sw = text_preprocessing.tokenize_text(text, remove_stopwords=True)
    tokenized_text_punct = text_preprocessing.tokenize_text(text, include_punctuation=True)
    phrases = text_preprocessing.split_into_phrases(text)
    feature_specs = {'no_of_words' : {'tokenized_text': tokenized_text},
                     'no_of_stop_words': {'tokenized_text' : tokenized_text, 'tokenized_text_sw' : tokenized_text_no_sw},
                     'no_of_contracted_wordforms': {'tokenized_text': tokenized_text_punct},
                     'no_of_characters': {'text': text},
                     'no_of_sentences': {'phrases' : phrases},
                     'avg_sentence_length': {'phrases': phrases, 'tokens': tokenized_text},
                     'no_of_punctuation': {'text': text},
                     'avg_word_length': {'tokenized_text': tokenized_text},
                     'lexical_diversity': {'tokenized_text': tokenized_text},
                     'unique_word_count': {'tokenized_text': tokenized_text}
                     }
    config = dataset.save_features(feature_specs=feature_specs)
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

def sentence_features(text):
    phrases = text_preprocessing.split_into_phrases(text)
    feature_specs = {'sentence_length_by_characters': {'sentences': phrases},
                     'sentence_length_by_word': {'sentences': phrases},
                     'sentence_avg_word_length': {'sentences': phrases},
                     'sentence_stopwords_count': {'sentences': phrases}}
    config = dataset.save_features(feature_specs=feature_specs)

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
    train_df.to_csv(f'Outputs/sentence_features.csv', index=False)

    return config, train_df


def tf_idf_features(text):
    tf_idf_with_sw = {
        'chapter' : text,
        'stop_words': False,
        'pos': False,
        'n_grams': False,
        'n': 2
    }
    tf_idf_without_sw = {
        'chapter': text,
        'stop_words': True,
        'pos': False,
        'n_grams': False,
        'n': 2
    }
    tf_idf_pos = {
        'chapter': text,
        'stop_words': False,
        'pos': True,
        'n_grams': False,
        'n': 2
    }
    tf_idf_bi_grams = {
        'chapter': text,
        'stop_words': False,
        'pos': False,
        'n_grams': True,
        'n': 2
    }
    tf_idf_tri_grams = {
        'chapter': text,
        'stop_words': False,
        'pos': False,
        'n_grams': True,
        'n': 3
    }

    feature_specs = {'tf_idf_feature': tf_idf_with_sw,
                     'tf_idf_feature': tf_idf_without_sw,
                     'tf_idf_feature': tf_idf_pos,
                     'tf_idf_feature': tf_idf_bi_grams,
                     'tf_idf_feature': tf_idf_tri_grams,
                     'tf_idf_punct': {'chapter': text},
                     'tf_idf_for_stopwords': {'chapter': text}}

    config = dataset.save_features(feature_specs=feature_specs)
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

def all_features(text):
    _, chapter_df = chapter_features(text)
    _, tf_idf_df = tf_idf_features(text)

    if 'label' not in chapter_df.columns or 'label' not in tf_idf_df.columns:
        raise ValueError("All DataFrames must contain a 'label' column")

    labels = tf_idf_df['label']

    df_chapter = chapter_df.drop(columns=['label'])
    df_tf_idf = tf_idf_df.drop(columns=['label'])

    df_chapter.index = labels.index
    df_tf_idf.index = labels.index

    combined_df = pd.concat([df_chapter, df_tf_idf], axis=1)

    combined_df['label'] = labels.values

    combined_df.fillna(0, inplace=True)
    combined_df = combined_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    combined_df.to_csv('Outputs/all_features.csv', index=False)
    return combined_df


def all_features_v2(sentence_path, chapter_path, tf_idf_path):
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
    combined_df.to_csv('Outputs/all_features_combined.csv', index=False)

    return combined_df

