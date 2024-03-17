import pandas as pd
import time_series
import text_preprocessing
import features
import configs
import classification

if __name__ == '__main__':

    chapters = text_preprocessing.split_into_chapters(text='New folder/Dos_Tol.txt', label = "DOS|TOL")

    lines = text_preprocessing.extract_lines(chapters)
    phrases = text_preprocessing.split_into_phrases(chapters)
    feature_phrases = features.sentence_length_by_word(phrases)

    output_file_path = 'Outputs/output.txt'
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for chapter_title in feature_phrases.items():
            output_file.write(f"{chapter_title}\n")

    feature = configs.chapter_features(chapters)

    data_path = f'Outputs/chapter_feature.csv'
    data_df = pd.read_csv(data_path)
    classification.classification(type = 'u', classifiers=['kmeans', 'pca'], data_df = data_df)

