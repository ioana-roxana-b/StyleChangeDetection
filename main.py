import pandas as pd

import text_preprocessing
import configs
import classification

if __name__ == '__main__':

    chapters = text_preprocessing.split_into_chapters(text='New folder/dickens.txt')
    phrases =text_preprocessing.split_into_phrases(chapters)
    #feature = configs.chapter_features(chapters)

    data_path = f'sentence_features.csv'
    data_df = pd.read_csv(data_path)
    X = data_df.drop('label', axis=1).values
    y = data_df['label'].apply(lambda x: x.split()[0]).values
    print(X)

    classification.classification(c=1, data_df=data_df, pc=True)
