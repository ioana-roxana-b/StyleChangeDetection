import copy
import re
import string
import pandas as pd
import nltk
from nltk import ngrams
from nltk.corpus import stopwords

import os

def split_into_chapters(dir = None, text = None, label = None):
    """
    Splits a large text into chapters based on chapter headers.
    Params:
        dir (str): Directory containing multiple text files (optional).
        text (str): Path to a single text file (optional).
        label (str): Custom label to search for in chapter headers (optional).
    Returns:
        dict: Dictionary with chapter headers as keys and chapter content as values.
    """
    if dir:
        text = ""
        # Loop through files in the directory
        for filename in os.listdir(dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir, filename)

                try:
                    # Read and concatenate file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        text += file.read() + "\n"
                except PermissionError:
                    print(f"Permission denied: {file_path}")

    elif text is not None:
        # If a single text file is provided, read its content
        all_content = ""
        with open(text, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        all_content += content + '\n'
        text = all_content
        #text = read_text(text)

    if isinstance(text, str):

        # Define pattern to identify chapter headers
        pattern = fr'\b(\d{{4}}|{label})\s+(CHAPTER|Chapter|PREFACE|Preface|EPILOGUE|Epilogue|Prologue|ACT|Act)[^\n]*'
        matches = list(re.finditer(pattern, text))
        chapters = {}

        # Split text into chapters based on identified headers
        for i, match in enumerate(matches):
            # Start of the chapter content
            start_idx = match.end()

            # End of the chapter content
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chapter_title = match.group(0).strip()
            chapter_text = text[start_idx:end_idx].strip()
            chapters[chapter_title] = chapter_text
    else:
        chapters = {}

    return chapters

def split_into_phrases(chapters):
    """
    Splits chapter content into individual phrases or sentences.
    Params:
        chapters (dict): Dictionary of chapters.
    Returns:
        dict: Updated dictionary with phrases for each chapter.
    """
    new_dict = {}

    for i in chapters.keys():
        # Split chapter content into sentences based on punctuation marks
        phrases = re.split(r'(?<=[.!?])\s+', chapters[i])

        # Remove extra whitespace and add to new dictionary
        phrases = [phrase.strip() for phrase in phrases if phrase.strip()]
        new_dict[i] = phrases

    return new_dict

def extract_lines(chapters):
    """
    Extracts only quoted lines from each chapter.
    Params:
        chapters (dict): Dictionary of chapters.
    Returns:
        dict: Dictionary with only quoted lines for each chapter.
    """
    updated_chapters = {}
    for chapter_number, text in chapters.items():

        # Use regex to find all quoted lines in the chapter
        quoted_texts = re.findall(r'(["])([A-Za-z].*?)\1(?=[\s.,?!"])', text, re.DOTALL)

        # Store extracted lines in the dictionary
        updated_chapters[chapter_number] = '\n'.join([match[1] for match in quoted_texts])

    return updated_chapters

def delete_punctuation_and_clean(text):
    """
    Removes punctuation and cleans up whitespace from the text.
    Params:
        text (dict): Dictionary of chapters or text segments.
    Returns:
        dict: Cleaned text without punctuation and unnecessary spaces.
    """
    for key in text.keys():
        # Remove punctuation and numbers
        text[key] = text[key].translate(str.maketrans('', '', string.punctuation + '0123456789'))

        # Replace multiple newlines with spaces
        text[key] = text[key].replace('\n\n', ' ').replace('\n', ' ')

        # Replace multiple spaces with a single space
        text[key] = re.sub(' +', ' ', text[key])

    return text

def lower_text(text_dict, include_punctuation = False):
    """
    Converts all text in the dictionary to lowercase.
    Params:
        text_dict (dict): Dictionary of text to be processed.
        include_punctuation (bool): Flag to include punctuation or not.
    Returns:
        dict: Text with all content converted to lowercase.
    """
    new_dict = copy.deepcopy(text_dict)
    for i in text_dict.keys():
        # Convert each segment of text to lowercase
        new_dict[i] = str.lower(new_dict[i])

    # If punctuation is excluded, perform additional cleaning
    if not include_punctuation:
        new_dict = delete_punctuation_and_clean(new_dict)

    return new_dict

def tokenize_text(text_dict, remove_stopwords = False, include_punctuation = False):
    """
    Tokenizes text content into individual words.
    Params:
        text_dict (dict): Dictionary of text to be tokenized.
        remove_stopwords (bool): Flag to remove stopwords or not.
        include_punctuation (bool): Flag to include punctuation or not.
    Returns:
        dict: Dictionary with tokenized content.
    """
    stop_words = set(stopwords.words('english'))
    new_dict = copy.deepcopy(text_dict)

    # Lowercase the text and clean if required
    text = lower_text(new_dict, include_punctuation)

    for key in text_dict.keys():
        # Tokenize the content into individual words
        tokens = nltk.word_tokenize(text[key])

        if remove_stopwords:
            # Remove stopwords if specified
            tokens = [word for word in tokens if word not in stop_words]

        new_dict[key] = tokens

    return new_dict

def pos_tag_text(text_dict):
    """
    Performs part-of-speech tagging on the tokenized text.
    Params:
        text_dict (dict): Dictionary of text to be tagged.
    Returns:
        dict: Dictionary with POS tags for each word.
    """
    new_dict = copy.deepcopy(text_dict)

    # Tokenize the text before tagging
    new_dict = tokenize_text(new_dict, remove_stopwords=False, include_punctuation=False)

    for key in text_dict.keys():
        # Apply POS tagging to each tokenized sentence
        new_dict[key] = nltk.pos_tag(new_dict[key])

    return new_dict

def create_vocab(chapters, stop_words = False, pos = False, n_grams = False, n = 2):
    """
    Creates a vocabulary of words, POS tags, or n-grams.
    Params:
        chapters (dict): Dictionary of chapters or text segments.
        stop_words (bool): Flag to exclude stopwords.
        pos (bool): Flag to use part-of-speech tags.
        n_grams (bool): Flag to create n-grams.
        n (int): Length of n-grams.
    Returns:
        list: Vocabulary list based on specified parameters.
    """
    vocab = []

    if pos:
        text_pos = pos_tag_text(chapters)
        for i in text_pos.keys():
            for j in text_pos[i]:
                word, pos = j
                if pos not in vocab:
                    vocab.append(pos)
        return vocab

    elif n_grams:
        # Create n-grams from the text
        text = tokenize_text(chapters, remove_stopwords=stop_words)
        for i in text.keys():
            for ng in ngrams(text[i], n):
                word = ' '.join(ng)
                if word not in vocab:
                    vocab.append(word)
        return vocab

    else:
        text = tokenize_text(chapters, remove_stopwords=stop_words)
        for i in text.keys():
            for word in text[i]:
                if word not in vocab:
                    vocab.append(word)
        return vocab

def extract_and_save_dialogues(quotation_info_path, output_file_path):
    """
    Extracts dialogues from a CSV file and saves them to a text file.
    Params:
        quotation_info_path (str): Path to the CSV file containing quotations.
        output_file_path (str): Path to save the extracted dialogues.
    Returns:
        dict: Dictionary containing dialogues for each character.
    """
    quotation_info = pd.read_csv(quotation_info_path)

    # Create a dictionary to store dialogues for each character
    dialogues = {}

    # Extract dialogues for each character in the CSV file
    for _, row in quotation_info.iterrows():
        character = row['speaker']
        quotation = row['quoteText']

        # Append dialogue to the corresponding character
        if character not in dialogues:
            dialogues[character] = ""

        dialogues[character] += quotation + " "

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save dialogues to a text file, annotated with character names
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for character, quotes in dialogues.items():
            output_file.write(f"{character} DIALOGUES\n")
            output_file.write(f"{quotes}\n\n")

    print(f"Dialogues saved to {output_file_path}")

    return dialogues
