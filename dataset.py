import os
import re
import string
import nltk
from nltk.corpus import stopwords

class TextProcessor:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def read_text(self):
        all_content = ""
        for filename in os.listdir(self.dir_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.dir_path, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    all_content += content + '\n'
        return all_content

    def split_into_chapters(self):
        text = self.read_text()
        sc = {}
        limits = r'\b(\d{4})\s+(CHAPTER|Chapter|PREFACE|Preface)\s*'
        lines = text.splitlines()
        list_of_acts = [l for l in lines if re.match(limits, l)]

        lim = '|'.join(map(re.escape, list_of_acts))
        scenes = re.split(lim, text)[1:]

        for (i, (j, scene)) in zip(range(len(list_of_acts)), enumerate(scenes)):
            sc[list_of_acts[i]] = scene.strip()

        return sc

    def split_into_phrases(self):
        chapters = self.split_into_chapters()
        for i in chapters.keys():
            phrases = re.split(r'[.!?]+', chapters[i])
            phrases = [phrase.strip() for phrase in phrases]
            chapters[i] = phrases
        return chapters

    def delete_punctuation(self):
        scenes = self.split_into_chapters()
        for i in scenes.keys():
            scenes[i] = scenes[i].translate(str.maketrans('', '', string.punctuation))
            scenes[i] = scenes[i].replace('\n\n', ' ')
        for i in scenes.keys():
            scenes[i] = re.sub(r'\d+', '', scenes[i])
        return scenes

    def lower_case_no_punct(self):
        scenes = self.delete_punctuation()
        for i in scenes.keys():
            scenes[i] = str.lower(scenes[i])
        return scenes

    def lower_case_with_punct(self):
        scenes = self.split_into_chapters()
        for i in scenes.keys():
            scenes[i] = str.lower(scenes[i])
        return scenes

    def text_tokenized_stopwords(self):
        scenes = self.lower_case_no_punct()
        for i in scenes.keys():
            scenes[i] = nltk.word_tokenize(scenes[i])
        return scenes

    def text_tokenized_no_stopwords(self):
        scenes = self.text_tokenized_stopwords()
        nltk_stopw = stopwords.words('english')
        ia_stopw = [line.strip() for line in open('stop_words')]
        for i in scenes.keys():
            scenes[i] = [word for word in scenes[i] if word not in ia_stopw]
        return scenes

    def text_tokenized_including_punctuation(self):
        scenes = self.split_into_chapters()
        for i in scenes.keys():
            scenes[i] = str.lower(scenes[i])
            scenes[i] = nltk.word_tokenize(scenes[i])
        return scenes

    def text_pos_tokenized_stopwords(self):
        scenes = self.text_tokenized_stopwords()
        for i in scenes.keys():
            scenes[i] = nltk.pos_tag(scenes[i])
        return scenes

