import os
import text_preprocessing
import features
import tf_idf_features
import dataset

if __name__ == '__main__':

    chapters = text_preprocessing.split_into_chapters(dir='Charles Dickens/Novels/1843_A Christmas Carol.txt')
    phrases = text_preprocessing.split_into_phrases(dir='New folder/Anna_Karenina.txt')
    lines = text_preprocessing.extract_lines(chapters)
    feature = dataset.save_chapter_features(lines,['tf_idf_feature'])
    #feature = features.no_of_characters(lines)

    output_file_path = 'output.txt'
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(f"{feature}\n")

    print(f"All text content written to {output_file_path}")
