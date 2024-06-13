import nltk
import pandas as pd
import text_preprocessing
import configs
import classification
def dialog():
    """""
    quotation_info_path = 'project-dialogism-novel-corpus-master/data/WinnieThePooh/quotation_info.csv'
    output_file_path = 'Outputs/dialogue.txt'

    dialogues = text_preprocessing.extract_and_save_dialogues(quotation_info_path, output_file_path)

    chap_feature = configs.chapter_features(dialogues)
    sent_feat = configs.sentence_features(dialogues)
    tf_idf_bigrams_features = configs.tf_idf_bigrams(dialogues)

    sentence_path = f'Outputs/sentence_features.csv'
    chapter_path = f'Outputs/chapter_feature.csv'
    tf_idf_path = f'Outputs/tf_idf_bigrams_features.csv'

    feature = configs.all_features_v3(sentence_path, tf_idf_path)
    """
    data_path = f'Outputs/WinnieThePooh/all_features_combined.csv'
    data_df = pd.read_csv(data_path)
    classification.classification(type='u', classifiers=['kmeans'], data_df=data_df,
                                  preprocessing_methods=['stand_sc'], dialog=True)

    #classification.classification(type='s', classifiers=['svm'], data_df=data_df, dialog=True)

def non_dialog():
    chapters = text_preprocessing.split_into_chapters(text='New folder/dickens_fict_nonfict.txt', label='DOS|TOL|FICTION|NONFICTION')

    configs.chapter_features(chapters)
    #configs.sentence_features(chapters)
    #configs.tf_idf_features(chapters)

   # sentence_path = f'Outputs/sentence_features.csv'
    #chapter_path = f'Outputs/chapter_feature.csv'
    #tf_idf_path = f'Outputs/tf_idf_features.csv'

   # configs.all_features_v2(sentence_path,chapter_path, tf_idf_path)

    data_path = f'Outputs/all_features_combined.csv.csv'
    data_df = pd.read_csv(data_path)
    classification.classification(type='s', classifiers=['random_forest'], data_df=data_df, preprocessing_methods=['minmax_sc'])
    classification.classification(type='u', classifiers=['kmeans'], data_df=data_df)


if __name__ == '__main__':
    """"OVERFITTING""
    data_path = f'Outputs/WinnieThePooh/bigrams.csv'
    data_df = pd.read_csv(data_path)
    classification.classification(type='s', classifiers=['random_forest'], data_df=data_df,
                                  preprocessing_methods=['pca'], dialog=True)
    """
    #non_dialog()
    dialog()



