import pandas as pd

from feature_configs import configs
from text_preprocessing import text_preprocessing

def extract_all_features(text_path, output_path, dialogue = False):
    """
    Extracts and computes multiple levels of text features (chapter, sentence, and TF-IDF) based on the provided configurations.
    Params:
        text_path (str): Path to the input text file.
        output_path (str): Directory where the resulting feature CSV files will be saved.
        dialogue (bool, optional): If True, extracts dialogues from the text and saves them to a file.
                                   Otherwise, splits the text into chapters.

    Returns:
            - chapter_df (pd.DataFrame): DataFrame containing chapter-level features.
            - sentence_df (pd.DataFrame): DataFrame containing sentence-level features.
            - tf_idf_df (pd.DataFrame): DataFrame containing TF-IDF features.
    """

    output_file_path = 'Outputs/dialogue.txt'

    chapter_config = 'chapter_features'
    chapter_config_file = 'feature_configs/chapter_configs.json'

    sentence_config = 'sentence_features'
    sentence_config_file = 'feature_configs/sentence_configs.json'

    tf_idf_config = 'tf_idf_features'
    tf_idf_config_file = 'feature_configs/tf_idf_configs.json'

    if dialogue:
        # Extract and save dialogues from the corpus to an output file
        text = text_preprocessing.extract_and_save_dialogues(text_path, output_file_path)
    else:
        text = text_preprocessing.split_into_chapters(text = text_path, label='Fletcher|Shakespeare|DOS|TOL|FICTION|NONFICTION')

    # Extract features using the configurations defined in `chapter_configs.json`
    chapter_df = configs.chapter_features(text = text, output_path = output_path, chapter_config = chapter_config, config_file = chapter_config_file)
    sentence_df = configs.sentence_features(text = text, output_path = output_path, sentence_config = sentence_config, config_file = sentence_config_file)
    tf_idf_df = configs.tf_idf_features(text = text, output_path = output_path, tf_idf_config = tf_idf_config, config_file = tf_idf_config_file)

    return chapter_df, sentence_df, tf_idf_df


def create_dfs(text_path, output_file_path, dialogue=False):
    """
       Creates and combines multiple feature DataFrames (sentence, chapter, and TF-IDF) for the input text.
       Params:
           text_path (str): Path to the input text file.
           output_file_path (str): Directory where the resulting feature CSV files will be saved.
           dialogue (bool, optional): If True, processes dialogues from the text; otherwise processes chapters.
       Returns:
           None: The function saves the resulting combined DataFrames to the output directory.
    """

    chapter_df, sentence_df, tf_idf_df = extract_all_features(text_path=text_path, output_path=output_file_path, dialogue=dialogue)

    # Combine features into a single DataFrame
    # configs.chapter_tf_idf(chapter_df = chapter_df, tf_idf_df = tf_idf_df, output_path = output_file_path)
    # configs.sentence_tf_idf(sentence_df = sentence_df, tf_idf_df = tf_idf_df, output_path = output_file_path)
    # configs.sentence_chapter(sentence_df = sentence_df, chapter_df = chapter_df, output_path = output_file_path)
    configs.all_features(sentence_df = sentence_df, chapter_df = chapter_df, tf_idf_df = tf_idf_df, output_path = output_file_path)

