import os
import re
import string

import nltk
from nltk.corpus import stopwords

def read_text(file_path):
    all_content = ""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    all_content += content + '\n'
    return all_content

def split_into_chapters(dir=None, text=None):
    if dir:
        text = read_text(dir)

    pattern = r'\b(\d{4})\s+(CHAPTER|Chapter|PREFACE|Preface|EPILOGUE|Epilogue)[^\n]*'
    matches = list(re.finditer(pattern, text))
    chapters = {}
    for i, match in enumerate(matches):
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapter_title = match.group(0).strip()
        chapter_text = text[start_idx:end_idx].strip()
        chapters[chapter_title] = chapter_text

    return chapters
def split_into_phrases(dir):
    chapters = split_into_chapters(dir)
    for i in chapters.keys():
        phrases = re.split(r'[.!?]+', chapters[i])
        phrases = [phrase.strip() for phrase in phrases]
        chapters[i] = phrases
    return chapters

def extract_lines(chapters):
    updated_chapters = {}
    for chapter_number, text in chapters.items():
        quoted_texts = re.findall(r'(["])([A-Za-z].*?)\1(?=[\s.,?!"])', text, re.DOTALL)
        updated_chapters[chapter_number] = [match[1] for match in quoted_texts]
    return updated_chapters

def delete_punctuation_and_clean(text):
    for key in text.keys():
        text[key] = text[key].translate(str.maketrans('', '', string.punctuation + '0123456789'))
        text[key] = text[key].replace('\n\n', ' ').replace('\n', ' ')
        text[key] = re.sub(' +', ' ', text[key])
    return text

def lower_text(text_dict, include_punctuation = False):
    for i in text_dict.keys():
        text_dict[i] = str.lower(text_dict[i])
    if include_punctuation == False:
        text_dict = delete_punctuation_and_clean(text_dict)
    return text_dict

def tokenize_text(text_dict, remove_stopwords=False, include_punctuation=False):
    stop_words = set(stopwords.words('english'))
    text = lower_text(text_dict, include_punctuation)
    for key in text_dict.keys():
        tokens = nltk.word_tokenize(text[key])
        if remove_stopwords:
            tokens = [word for word in tokens if word not in stop_words]
        text_dict[key] = tokens
    return text_dict

def pos_tag_text(text_dict):
    text_dict = tokenize_text(text_dict, remove_stopwords=False, include_punctuation=False)
    for key in text_dict.keys():
        text_dict[key] = nltk.pos_tag(text_dict[key])
    return text_dict

