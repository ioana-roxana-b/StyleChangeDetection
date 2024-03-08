import os
import dataset
import features
import tf_idf_features

if __name__ == '__main__':
    """""
    directory_path = 'Charles Dickens/Novels'
    all_text_content = ""
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            text = dataset.read_text(filepath)
            all_text_content += text
    """

    chapters = dataset.split_into_chapters(dir='Charles Dickens/Novels/1843_A Christmas Carol.txt')
    phrases = dataset.split_into_phrases(dir='New folder/Anna_Karenina.txt')
    lines = dataset.tokenize_text(chapters, remove_stopwords=False, include_punctuation=False)

    feature = features.lexical_diversity(lines)

    output_file_path = 'output.txt'
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(f"{feature}\n")

    print(f"All text content written to {output_file_path}")
