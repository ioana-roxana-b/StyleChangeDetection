import nltk
import pandas as pd
import text_preprocessing
import configs
import classification
def dialog():
    """""
    quotation_info_path = 'project-dialogism-novel-corpus-master/data/Emma/quotation_info.csv'
    output_file_path = 'Outputs/dialogue.txt'

    dialogues = text_preprocessing.extract_and_save_dialogues(quotation_info_path, output_file_path)

    configs.chapter_features(dialogues)
    configs.sentence_features(dialogues)
    configs.tf_idf_bigrams(dialogues)

    sentence_path = f'Outputs/WinnieThePooh/sentence_features.csv'
    chapter_path = f'Outputs/WinnieThePooh/chapter_feature.csv'
    tf_idf_path = f'Outputs/WinnieThePooh/tf_idf_features.csv'

    configs.all_features_v2(sentence_path, chapter_path, tf_idf_path)
    """""
    data_path = f'Outputs/Emma/all_features_sent.csv'
    data_df = pd.read_csv(data_path)


    classification.classification(type='s', classifiers=['knn'], data_df=data_df,
                                  preprocessing_methods=[ 'pca', 'stand_sc'], dialog=True)
    classification.classification(type='s', classifiers=['random_forest'], data_df=data_df,
                                  preprocessing_methods=['pca','stand_sc'], dialog=True)
    classification.classification(type='s', classifiers=['svm'], data_df=data_df,
                                  preprocessing_methods=['pca', 'stand_sc'],  dialog=True)
   # classification.classification(type='u', classifiers=['kmeans'], data_df=data_df,
    #                              preprocessing_methods=['pca', 'minmax_sc'], dialog=True)

def non_dialog():
    #chapters = text_preprocessing.split_into_chapters(text='New folder/dickens.txt', label='DOS|TOL|FICTION|NONFICTION')

    #configs.chapter_features(chapters)
    #configs.sentence_features(chapters)
    #configs.tf_idf_features(chapters)

    #sentence_path = f'Outputs/Fict_Nonfict/sentence_features.csv'
    #chapter_path = f'Outputs/Fict_Nonfict/chapter_feature.csv'
    #tf_idf_path = f'Outputs/Fict_Nonfict/tf_idf_features.csv'

    #configs.all_features_v2(sentence_path,chapter_path, tf_idf_path)

    data_path = (f'Outputs/Fict_Nonfict/tf_idf_features.csv')
    data_df = pd.read_csv(data_path)
    #classification.classification(type='s', classifiers=['knn'], data_df=data_df)
   # classification.classification(type='s', classifiers=['random_forest'], data_df=data_df)
    #classification.classification(type='s', classifiers=['svm'], data_df=data_df)
    classification.classification(type='u', classifiers=['kmeans'], data_df=data_df, preprocessing_methods=['lasso'])


if __name__ == '__main__':
    """"OVERFITTING""
    data_path = f'Outputs/WinnieThePooh/bigrams.csv'
    data_df = pd.read_csv(data_path)
    classification.classification(type='s', classifiers=['random_forest'], data_df=data_df,
                                  preprocessing_methods=['pca'], dialog=True)
    """
    non_dialog()
    #dialog()



