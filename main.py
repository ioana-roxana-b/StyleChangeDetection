import nltk
import pandas as pd
import text_preprocessing
import configs
import classification
if __name__ == '__main__':

    """""""""
    chapters = text_preprocessing.split_into_chapters(text='New folder/Dos_Tol.txt', label='DOS|TOL')
    sentence = text_preprocessing.split_into_phrases(chapters)

    output_file_path = 'Outputs/output.txt'
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for chapter_title in sentence.items():
            output_file.write(f"{chapter_title}\n")

    sent_feature = configs.sentence_features(chapters)
    chap_feature = configs.chapter_features(chapters)
    tf_ifd_feature = configs.tf_idf_features(chapters)

    sentence_path = f'Outputs/sentence_features.csv'
    chapter_path = f'Outputs/chapter_feature.csv'
    tf_idf_path = f'Outputs/tf_idf_features.csv'

    feature = configs.all_features_v2(chapters)

    data_path = f'Outputs/all_features_v2.csv'
    data_df = pd.read_csv(data_path)
    classification.classification(type='u', classifiers=['kmeans'], data_df=data_df)

    novel_text_path = 'project-dialogism-novel-corpus-master/data/Emma/novel_text.txt'
    character_info_path = 'project-dialogism-novel-corpus-master/data/Emma/character_info.csv'
    quotation_info_path = 'project-dialogism-novel-corpus-master/data/Emma/quotation_info.csv'

    with open(novel_text_path, 'r', encoding='utf-8') as file:
        novel_text = file.read()


    output_file_path = 'Outputs/dialogue.txt'
    # Extract and save dialogues
    dialogues = text_preprocessing.extract_and_save_dialogues(quotation_info_path, character_info_path,
                                                              output_file_path)

    sent = text_preprocessing.split_into_phrases(dialogues)
    output_file_path = 'Outputs/output.txt'
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for chapter_title in sent.items():
            output_file.write(f"{chapter_title}\n")

    sentence_path = f'Outputs/sentence_features.csv'
    chapter_path = f'Outputs/chapter_feature.csv'
    tf_idf_path = f'Outputs/tf_idf_features.csv'

    feature = configs.all_features_v2(sentence_path, chapter_path, tf_idf_path)
"""""""""
    data_path = f'Outputs/all_features_combined.csv'
    data_df = pd.read_csv(data_path)
    classification.classification(type='u', classifiers=['kmeans'], data_df=data_df, preprocessing_methods=['minmax_sc'], dialog=True)


