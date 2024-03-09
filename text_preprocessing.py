import copy
import os
import string

import nltk
from nltk import ngrams
from nltk.corpus import stopwords

import os
import re

def read_text(file_path):
    all_content = ""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    all_content += content + '\n'
    return all_content

def split_into_chapters(dir=None, text=None):
    if dir:
        text = ""
        for filename in os.listdir(dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        text += file.read() + "\n"
                except PermissionError:
                    print(f"Permission denied: {file_path}")
    elif text is not None:
        text = read_text(text)

    if isinstance(text, str):
        pattern = r'\b(\d{4}|Shakespeare|Fletcher|DOS|TOL)\s+(CHAPTER|Chapter|PREFACE|Preface|EPILOGUE|Epilogue|Prologue|ACT|Act)[^\n]*'
        matches = list(re.finditer(pattern, text))
        chapters = {}
        for i, match in enumerate(matches):
            start_idx = match.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chapter_title = match.group(0).strip()
            chapter_text = text[start_idx:end_idx].strip()
            chapters[chapter_title] = chapter_text
    else:
        chapters = {}

    return chapters

def split_into_phrases(chapters):
    for i in chapters.keys():
        phrases = re.split(r'[.!?]+', chapters[i])
        phrases = [phrase.strip() for phrase in phrases]
        chapters[i] = phrases
    return chapters

import re

def extract_lines(chapters):
    updated_chapters = {}
    for chapter_number, text in chapters.items():
        quoted_texts = re.findall(r'(["])([A-Za-z].*?)\1(?=[\s.,?!"])', text, re.DOTALL)
        updated_chapters[chapter_number] = '\n'.join([match[1] for match in quoted_texts])
    return updated_chapters


def delete_punctuation_and_clean(text):
    for key in text.keys():
        text[key] = text[key].translate(str.maketrans('', '', string.punctuation + '0123456789'))
        text[key] = text[key].replace('\n\n', ' ').replace('\n', ' ')
        text[key] = re.sub(' +', ' ', text[key])
    return text

def lower_text(text_dict, include_punctuation = False):
    new_dict = copy.deepcopy(text_dict)
    for i in text_dict.keys():
        new_dict[i] = str.lower(new_dict[i])
    if include_punctuation == False:
        new_dict = delete_punctuation_and_clean(new_dict)
    return new_dict

def tokenize_text(text_dict, remove_stopwords=False, include_punctuation=False):
    stop_words = set(stopwords.words('english'))
    new_dict = copy.deepcopy(text_dict)
    text = lower_text(new_dict, include_punctuation)
    for key in text_dict.keys():
        tokens = nltk.word_tokenize(text[key])
        if remove_stopwords:
            tokens = [word for word in tokens if word not in stop_words]
        new_dict[key] = tokens
    return new_dict

def pos_tag_text(text_dict):
    new_dict = copy.deepcopy(text_dict)
    new_dict = tokenize_text(new_dict, remove_stopwords=False, include_punctuation=False)
    for key in text_dict.keys():
        new_dict[key] = nltk.pos_tag(new_dict[key])
    return new_dict

def create_vocab(chapters, stop_words=False, pos=False, n_grams=False, n=2):
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
