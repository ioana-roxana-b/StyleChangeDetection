import text_preprocessing
import configs

if __name__ == '__main__':

    chapters = text_preprocessing.split_into_chapters(text='New folder/Anna_Karenina.txt')
    #phrases = text_preprocessing.split_into_phrases(dir='New folder/Anna_Karenina.txt')
    feature = configs.chapter_features(chapters)

