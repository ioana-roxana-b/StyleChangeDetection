import os
import dataset
import features

if __name__ == '__main__':

    directory_path = 'Charles Dickens/Novels'
    all_text_content = ""
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            text = dataset.read_text(filepath)
            all_text_content += text

    chapters = dataset.split_into_chapters(dir='New folder/Crime and Punishment.txt')
    #lines = dataset.tokenize_text(chapters, remove_stopwords=False, include_punctuation=False)
    feature = features.no_of_characters(chapters)

    output_file_path = 'output.txt'
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for chapter_title in feature.items():
            output_file.write(f"{chapter_title}\n")

    print(f"All text content written to {output_file_path}")
