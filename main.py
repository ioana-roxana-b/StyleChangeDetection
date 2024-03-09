import pandas as pd

import text_preprocessing
import configs
import classification

if __name__ == '__main__':

    chapters = text_preprocessing.split_into_chapters(text='New folder/Dos_Tol_1.txt')
    feature = configs.chapter_features(chapters)

    data_path = f'chapter_feature.csv'
    data_df = pd.read_csv(data_path)

    classification.classification(c=1, data_df=data_df, pc=True)
