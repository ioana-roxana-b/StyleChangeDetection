import string
import math
import numpy as np
from nltk import ngrams
from nltk.corpus import stopwords
import text_preprocessing

def term_frequency(word, scene):
    words = scene.split()
    return words.count(word)

def inverse_document_frequency(word, scenes):
    num_scenes_with_word = sum(1 for scene in scenes.values() if word in scene)
    if num_scenes_with_word == 0:
        return 0
    else:
        return math.log(len(scenes) / num_scenes_with_word)

def n_grams_tf(ngram, scene):
    words = list(ngrams(scene.split(), len(ngram.split())))
    words_str = [' '.join(word) for word in words]
    return words_str.count(ngram)

def n_grams_idf(ngram, scenes):
    num_scenes_with_ngram = sum(1 for scene in scenes.values() if ngram in scene)
    if num_scenes_with_ngram == 0:
        return 0
    else:
        return math.log(len(scenes) / num_scenes_with_ngram)

def pos_tf(pos, scene):
    count = 0
    for j in scene:
        word, tag =j
        if tag==pos:
            count +=1
    return count

def pos_idf(pos, scenes):
    num_scenes_with_pos = 0
    for i in scenes.keys():
        for j in scenes[i]:
            word, tag = j
            if tag == pos:
                num_scenes_with_pos +=1

    if num_scenes_with_pos == 0:
        return 0
    else:
        return math.log(len(scenes) / num_scenes_with_pos)

def tf_idf_feature(chapter, stop_words=False, pos = False, n_grams=False, n=2):
    text = text_preprocessing.lower_text(text_dict=chapter, include_punctuation=False)
    if pos:
        pos_set = text_preprocessing.create_vocab(text, stop_words=stop_words, pos=True)
        tokens = text_preprocessing.pos_tag_text(text)
        pos_index = {}
        for i, pos in enumerate(pos_set):
            pos_index[pos] = i

        tf_idf_matrix = np.zeros((len(tokens.keys()), len(pos_set)))

        for (i, j) in zip(tokens.keys(), range(len(tokens))):
            vec = np.zeros((len(pos_set),))
            for z in tokens[i]:
                word, pos = z
                tf = pos_tf(pos, tokens[i])
                idf = pos_idf(pos, tokens)
                vec[pos_index[pos]] = tf * idf
            tf_idf_matrix[j] = vec

    elif n_grams:
        ngrams_set = text_preprocessing.create_vocab(text, n_grams=True, n=n)
        tokens = text_preprocessing.tokenize_text(text, remove_stopwords=stop_words)
        word_index = {}
        for i, word in enumerate(ngrams_set):
            word_index[word] = i

        tf_idf_matrix = np.zeros((len(tokens.keys()), len(ngrams_set)))

        for (i, j) in zip(text.keys(), range(len(text))):
            vec = np.zeros((len(ngrams_set),))
            for ng in ngrams(tokens[i], n):
                word = ' '.join(ng)
                tf = n_grams_tf(word, text[i])
                idf = n_grams_idf(word, text)
                vec[word_index[word]] = tf * idf
            tf_idf_matrix[j] = vec
    else:
        word_set = text_preprocessing.create_vocab(text, stop_words=stop_words)
        tokens = text_preprocessing.tokenize_text(text, remove_stopwords=stop_words)
        word_index = {}
        for i, word in enumerate(word_set):
            word_index[word] = i

        tf_idf_matrix = np.zeros((len(tokens.keys()), len(word_set)))

        for (i,j) in zip(text.keys(), range(len(text))):
            vec = np.zeros((len(word_set),))
            for word in tokens[i]:
                tf = term_frequency(word, text[i])
                idf = inverse_document_frequency(word, text)
                vec[word_index[word]] = tf * idf
            tf_idf_matrix[j] = vec

    for (i,j) in zip(text.keys(),range(len(text))):
        text[i] = tf_idf_matrix[j]
    return text

def tf_idf_for_stopwords(chapter):
    text = text_preprocessing.lower_text(text_dict=chapter, include_punctuation=False)
    word_set = stopwords.words('english')
    word_index = {}
    for i, word in enumerate(word_set):
        word_index[word] = i

    tf_idf_matrix = np.zeros((len(text.keys()), len(word_set)))

    for (i,j) in zip(text.keys(), range(len(text))):
        vec = np.zeros((len(word_set),))
        for word in word_set:
            tf = term_frequency(word, text[i])
            idf = inverse_document_frequency(word, text)
            vec[word_index[word]] = tf * idf
        tf_idf_matrix[j] = vec

    for (i, j) in zip(text.keys(), range(len(text))):
        text[i] = tf_idf_matrix[j]
    return text

def punct_tf(char, scene):
    return scene.count(char)

def punct_idf(char, scenes):
    num_scenes_with_char = sum(1 for scene in scenes.values() if char in scene)

    if num_scenes_with_char == 0:
        return 0
    else:
        return math.log(len(scenes) / num_scenes_with_char)
def tf_idf_punct(chapter):
    text = text_preprocessing.lower_text(chapter, include_punctuation=True)
    word_set = string.punctuation
    word_index = {}
    for i, word in enumerate(word_set):
        word_index[word] = i

    tf_idf_matrix = np.zeros((len(text.keys()), len(word_set)))

    for (i, j) in zip(text.keys(), range(len(text))):
        vec = np.zeros((len(word_set),))
        for word in word_set:
            tf = punct_tf(word, text[i])
            idf = punct_idf(word, text)
            vec[word_index[word]] = tf * idf
        tf_idf_matrix[j] = vec

    for (i, j) in zip(text.keys(), range(len(text))):
        text[i] = tf_idf_matrix[j]
    return text


