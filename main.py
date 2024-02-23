import dataset
if __name__ == '__main__':
    directory_path = 'Charles Dickens/Novels'

    processor = dataset.TextProcessor(directory_path)
    all_text_content = processor.read_text()
    chapters = processor.split_into_chapters()
    sentences = processor.split_into_phrases()

    # Write the combined content to a new text file
    output_file_path = 'output.txt'
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for chapter_title in sentences.keys():
            output_file.write(f"{chapter_title}\n")

    print(f"All text content written to {output_file_path}")
