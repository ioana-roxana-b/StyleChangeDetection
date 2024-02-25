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

def split_into_chapters(dir):
    text = read_text(dir)
    sc = {}
    limits = r'\b(\d{4})\s+(CHAPTER|Chapter|PREFACE|Preface)\s*'
    lines = text.splitlines()
    list_of_acts = [l for l in lines if re.match(limits, l)]

    lim = '|'.join(map(re.escape, list_of_acts))
    text = re.split(lim, text)[1:]

    for (i, (j, scene)) in zip(range(len(list_of_acts)), enumerate(text)):
        sc[list_of_acts[i]] = scene.strip()

    return sc

def split_into_phrases(dir):
    chapters = split_into_chapters(dir)
    for i in chapters.keys():
        phrases = re.split(r'[.!?]+', chapters[i])
        phrases = [phrase.strip() for phrase in phrases]
        chapters[i] = phrases
    return chapters

def extract_lines(dir):
    chapters = split_into_chapters(dir)
    updated_chapters = {}
    for chapter_number, text in chapters.items():
        quoted_texts = re.findall(r'(["])([A-Za-z].*?)\1(?=[\s.,?!"])', text, re.DOTALL)
        updated_chapters[chapter_number] = [match[1] for match in quoted_texts]

    print(updated_chapters)

    return updated_chapters

def delete_punctuation(text):
    for i in text.keys():
        text[i] = text[i].translate(str.maketrans('', '', string.punctuation))
        text[i] = text[i].replace('\n\n', ' ')
    for i in text.keys():
        text[i] = re.sub(r'\d+', '', text[i])
    return text

def lower_case_no_punct(dir):
    text = delete_punctuation(dir)
    for i in text.keys():
        text[i] = str.lower(text[i])
    return text

def lower_case_with_punct(text):
    for i in text.keys():
        text[i] = str.lower(text[i])
    return text

def text_tokenized_stopwords(text):
    for i in text.keys():
        text[i] = nltk.word_tokenize(text[i])
    return text

def text_tokenized_no_stopwords(text):
    text = text_tokenized_stopwords(text)
    nltk_stopw = stopwords.words('english')
    ia_stopw = [line.strip() for line in open('stop_words')]
    for i in text.keys():
        for j in text[i]:
            if j in ia_stopw:
                text[i].remove(j)
    return text

def text_tokenized_including_punctuation(text):
    for i in text.keys():
        text[i] = str.lower(text[i])
        text[i] = nltk.word_tokenize(text[i])
    return text

def text_pos_tokenized_stopwords(text):
    text = text_tokenized_stopwords(text)
    for i in text.keys():
        text[i] = nltk.pos_tag(text[i])
    return text
