import string
import math
import numpy as np
from nltk import ngrams
from nltk.corpus import stopwords
from text_preprocessing import text_preprocessing


def term_frequency(word, text):
    """
    Calculates the term frequency (TF) of a word in a given text.
    Params:
        word (str): The word to calculate TF for.
        text (str): The text in which to calculate the TF.
    Returns:
        int: Frequency of the word in the text.
    """
    words = text.split()
    return words.count(word)


def inverse_document_frequency(word, texts):
    """
    Calculates the inverse document frequency (IDF) of a word across texts.
    Params:
        word (str): The word to calculate IDF for.
        texts (dict): Dictionary of texts.
    Returns:
        float: IDF score of the word.
    """
    # Count how many texts contain the word
    num_texts_with_word = sum(1 for text in texts.values() if word in text)

    if num_texts_with_word == 0:
        return 0

    else:
        return math.log(len(texts) / num_texts_with_word)


def n_grams_tf(ngram, text):
    """
    Calculates the term frequency (TF) of an n-gram in a given text.
    Params:
        ngram (str): The n-gram to calculate TF for.
        text (str): The text in which to calculate the TF.
    Returns:
        int: Frequency of the n-gram in the text.
    """
    # Split the text into n-grams
    words = list(ngrams(text.split(), len(ngram.split())))
    words_str = [' '.join(word) for word in words]

    return words_str.count(ngram)


def n_grams_idf(ngram, texts):
    """
    Calculates the inverse document frequency (IDF) of an n-gram across texts.
    Params:
        ngram (str): The n-gram to calculate IDF for.
        texts (dict): Dictionary of texts.
    Returns:
        float: IDF score of the n-gram.
    """
    # Count how many texts contain the n-gram
    num_texts_with_ngram = sum(1 for text in texts.values() if ngram in text)

    if num_texts_with_ngram == 0:
        return 0

    else:
        return math.log(len(texts) / num_texts_with_ngram)


def pos_tf(pos, text):
    """
    Calculates the term frequency (TF) of a part-of-speech (POS) tag in a text.
    Params:
        pos (str): The POS tag to calculate TF for.
        text (list): List of tuples containing (word, POS) for the text.
    Returns:
        int: Frequency of the POS tag in the text.
    """
    count = 0
    for j in text:
        word, tag = j
        if tag == pos:
            count += 1

    return count


def pos_idf(pos, texts):
    """
    Calculates the inverse document frequency (IDF) of a part-of-speech (POS) tag across texts.
    Params:
        pos (str): The POS tag to calculate IDF for.
        texts (dict): Dictionary of texts.
    Returns:
        float: IDF score of the POS tag.
    """
    num_texts_with_pos = 0

    # Count the number of texts containing the POS tag
    for i in texts.keys():
        for j in texts[i]:
            word, tag = j
            if tag == pos:
                num_texts_with_pos += 1

    if num_texts_with_pos == 0:
        return 0

    else:
        return math.log(len(texts) / num_texts_with_pos)


def tf_idf_feature(chapter, stop_words = False, pos = False, n_grams = False, n = 2):
    """
    Calculates the TF-IDF features for text based on words, POS tags, or n-grams.
    Params:
        chapter (dict): Dictionary containing the chapter text.
        stop_words (bool): Whether to exclude stopwords in the calculation.
        pos (bool): Whether to use POS tags for TF-IDF calculation.
        n_grams (bool): Whether to use n-grams for TF-IDF calculation.
        n (int): Length of n-grams if using n-grams.
    Returns:
        dict: Dictionary of chapters with TF-IDF scores.
    """
    text = text_preprocessing.lower_text(text_dict=chapter, include_punctuation=False)

    if pos:

        # Calculate TF-IDF using POS tags
        pos_set = text_preprocessing.create_vocab(text, stop_words=stop_words, pos=True)
        tokens = text_preprocessing.pos_tag_text(text)
        pos_index = {pos: i for i, pos in enumerate(pos_set)}

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

        # Calculate TF-IDF using n-grams
        ngrams_set = text_preprocessing.create_vocab(text, n_grams=True, n=n)
        tokens = text_preprocessing.tokenize_text(text, remove_stopwords=stop_words)
        word_index = {word: i for i, word in enumerate(ngrams_set)}

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

        # Calculate TF-IDF using individual words
        word_set = text_preprocessing.create_vocab(text, stop_words=stop_words)
        tokens = text_preprocessing.tokenize_text(text, remove_stopwords=stop_words)
        word_index = {word: i for i, word in enumerate(word_set)}

        tf_idf_matrix = np.zeros((len(tokens.keys()), len(word_set)))

        for (i, j) in zip(text.keys(), range(len(text))):
            vec = np.zeros((len(word_set),))
            for word in tokens[i]:
                tf = term_frequency(word, text[i])
                idf = inverse_document_frequency(word, text)
                vec[word_index[word]] = tf * idf
            tf_idf_matrix[j] = vec

    # Update the chapters with their corresponding TF-IDF vectors
    for (i, j) in zip(text.keys(), range(len(text))):
        text[i] = tf_idf_matrix[j]

    return text


def tf_idf_for_stopwords(chapter):
    """
    Calculates TF-IDF for stopwords across the chapter.
    Params:
        chapter (dict): Dictionary containing the chapter text.
    Returns:
        dict: Dictionary of chapters with TF-IDF scores for stopwords.
    """

    text = text_preprocessing.lower_text(text_dict=chapter, include_punctuation=False)
    word_set = stopwords.words('english')
    word_index = {word: i for i, word in enumerate(word_set)}

    tf_idf_matrix = np.zeros((len(text.keys()), len(word_set)))

    for (i, j) in zip(text.keys(), range(len(text))):
        vec = np.zeros((len(word_set),))
        for word in word_set:
            tf = term_frequency(word, text[i])
            idf = inverse_document_frequency(word, text)
            vec[word_index[word]] = tf * idf
        tf_idf_matrix[j] = vec

    for (i, j) in zip(text.keys(), range(len(text))):
        text[i] = tf_idf_matrix[j]

    return text


def punct_tf(char, text):
    """
    Calculates the term frequency (TF) of a punctuation mark in a given text.
    Params:
        char (str): The punctuation character to calculate TF for.
        text (str): The text in which to calculate the TF.
    Returns:
        int: Frequency of the punctuation mark in the text.
    """
    return text.count(char)


def punct_idf(char, texts):
    """
    Calculates the inverse document frequency (IDF) of a punctuation mark across texts.
    Params:
        char (str): The punctuation character to calculate IDF for.
        texts (dict): Dictionary of texts.
    Returns:
        float: IDF score of the punctuation mark.
    """
    num_texts_with_char = sum(1 for text in texts.values() if char in text)

    if num_texts_with_char == 0:
        return 0

    else:
        return math.log(len(texts) / num_texts_with_char)


def tf_idf_punct(chapter):
    """
    Calculates TF-IDF for punctuation marks across the chapter.
    Params:
        chapter (dict): Dictionary containing the chapter text.
    Returns:
        dict: Dictionary of chapters with TF-IDF scores for punctuation marks.
    """
    text = text_preprocessing.lower_text(chapter, include_punctuation=True)
    word_set = string.punctuation
    word_index = {word: i for i, word in enumerate(word_set)}

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
