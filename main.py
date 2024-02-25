import os

import dataset
import features
import sentence_fatures
if __name__ == '__main__':

    directory_path = 'Charles Dickens/Novels'
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            all_text_content = dataset.read_text(filepath)

    lines = dataset.extract_lines('Charles Dickens/Novels/1843_A Christmas Carol.txt')

    output_file_path = 'output.txt'
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for chapter_title in lines.items():
            output_file.write(f"{chapter_title}\n")

    print(f"All text content written to {output_file_path}")

    #features.no_of_words(directory_path)
    #sentence_fatures.sentence_length_by_word(directory_path)

