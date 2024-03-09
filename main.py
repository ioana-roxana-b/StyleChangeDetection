import copy
import os
import text_preprocessing
import features
import tf_idf_features
import dataset
import configs

if __name__ == '__main__':

    chapters = text_preprocessing.split_into_chapters(dir='Charles Dickens/Novels/1843_A Christmas Carol.txt')
    #phrases = text_preprocessing.split_into_phrases(dir='New folder/Anna_Karenina.txt')
    feature = configs.tf_idf_features(chapters)

