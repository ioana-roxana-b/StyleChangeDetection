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

    feature = configs.sentence_features(chapters)

    data_path = f'Outputs/sentence_features.csv'
    data_df = pd.read_csv(data_path)
    classification.classification(type='u', classifiers=['kmeans'], data_df=data_df)

"""""""""

    novel_text_path = 'project-dialogism-novel-corpus-master/data/AnneOfGreenGables/novel_text.txt'
    character_info_path = 'project-dialogism-novel-corpus-master/data/AnneOfGreenGables/character_info.csv'
    quotation_info_path = 'project-dialogism-novel-corpus-master/data/AnneOfGreenGables/quotation_info.csv'

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

    feature = configs.sentence_features(dialogues)

    data_path = f'Outputs/sentence_features.csv'
    data_df = pd.read_csv(data_path)
    classification.classification(type='u', classifiers=['kmeans'], data_df=data_df, dialog=True)



