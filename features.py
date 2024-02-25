import string
import dataset
import re

def no_of_words(dir):
    chapters = dataset.text_tokenized_stopwords(dir)
    no_of_w = {}
    for i in chapters.keys():
        no_of_w[i] = len(chapters[i])
    return no_of_w

def no_of_stop_words(dir):
    chapters = dataset.text_tokenized_stopwords(dir)
    chapters_no_sw = dataset.text_tokenized_no_stopwords(dir)
    no_of_sw = {}
    for (i, j) in zip(chapters.keys(), chapters_no_sw.keys()):
        no_of_sw[i] = len(chapters[i]) - len(chapters_no_sw[j])
    return no_of_sw

def no_of_contracted_wordforms(dir):
    chapters = dataset.split_into_chapters(dir)
    pattern = r"\b\w+'\w+\b"
    for i in chapters.keys():
        contracted_word_forms = re.findall(pattern, chapters[i])
        num_contracted_word_forms = len(contracted_word_forms)
        chapters[i]=num_contracted_word_forms
    return chapters

def no_of_characters(dir):
    chapters = dataset.split_into_chapters(dir)
    no_of_ch = {}
    for i in chapters.keys():
        no_of_ch[i] = len(chapters[i])
    return no_of_ch

def no_of_sentences(dir):
    chapters = dataset.split_chapters_into_phrases(dir)
    no_of_s = {}
    for i in chapters.keys():
        no_of_s[i] = len(chapters[i])
    return no_of_s

def avg_sentence_length(dir):
    phrases = dataset.split_chapters_into_phrases(dir)
    words = dataset.text_tokenized_stopwords(dir)
    for i in phrases.keys():
        avg_sentence_len = len(words[i]) / len(phrases[i])
        phrases[i] = avg_sentence_len
    return phrases

def no_of_punctuation(dir):
    chapters = dataset.split_into_chapters(dir)
    no_of_ch = {}
    for i in chapters.keys():
        nr_p = 0
        for j in chapters[i]:
            if j in string.punctuation:
                nr_p +=1
        no_of_ch[i] = nr_p
    return no_of_ch

def scene_avg_word_length(dir):
    chapters = dataset.delete_punctuation(dir)
    no_of_ch = {}
    for i in chapters.keys():
        len(chapters[i])
        no_of_ch[i] = len(chapters[i])

    words = dataset.text_tokenized_stopwords(dir)
    for i in words.keys():
        avg_word_len = len(words[i])/(no_of_ch[i])
        words[i]=avg_word_len
    return words


