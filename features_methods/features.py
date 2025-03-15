import string
from nltk.corpus import stopwords
import re

def no_of_words(tokenized_text):
    """
    Calculates the number of words in each chapter.
    Params:
        tokenized_text (dict): Dictionary with tokenized words for each chapter.
    Returns:
        dict: Number of words in each chapter.
    """
    no_of_w = {chapter: len(tokens) for chapter, tokens in tokenized_text.items()}
    return no_of_w

def no_of_stop_words(tokenized_text, tokenized_text_sw):
    """
    Calculates the number of stopwords in each chapter.
    Params:
        tokenized_text (dict): Original tokenized text for each chapter.
        tokenized_text_sw (dict): Tokenized text with stopwords removed for each chapter.
    Returns:
        dict: Number of stopwords in each chapter.
    """
    no_of_sw = {}
    # Calculate the number of stopwords by comparing the lengths of the original and stopword-removed texts
    for (i, j) in zip(tokenized_text.keys(), tokenized_text_sw.keys()):
        no_of_sw[i] = len(tokenized_text[i]) - len(tokenized_text_sw[j])

    return no_of_sw

def no_of_contracted_wordforms(tokenized_text):
    """
    Counts the number of contracted word forms (e.g., can't, won't) in each chapter.
    Params:
        tokenized_text (dict): Dictionary with tokenized words for each chapter.
    Returns:
        dict: Number of contracted word forms in each chapter.
    """

    # Pattern to identify contractions
    pattern = r"\b\w+'\w+\b"
    new_dict = {}
    for i in tokenized_text.keys():
        # Join tokens into a single string
        text = ' '.join(tokenized_text[i])
        # Find contracted word forms
        contracted_word_forms = re.findall(pattern, text)
        num_contracted_word_forms = len(contracted_word_forms)
        new_dict[i] = num_contracted_word_forms

    return new_dict

def no_of_characters(text):
    """
    Calculates the number of characters in each chapter.
    Params:
        text (dict): Dictionary of text segments for each chapter.
    Returns:
        dict: Number of characters in each chapter.
    """
    no_of_ch = {}
    for i in text.keys():
        no_of_ch[i] = len(text[i])
    return no_of_ch

def no_of_sentences(phrases):
    """
    Counts the number of sentences in each chapter.
    Params:
        phrases (dict): Dictionary containing sentences for each chapter.
    Returns:
        dict: Number of sentences in each chapter.
    """
    no_of_s = {}
    for i in phrases.keys():
        no_of_s[i] = len(phrases[i])
    return no_of_s

def avg_sentence_length(phrases, tokens):
    """
    Calculates the average sentence length in terms of words.
    Params:
        phrases (dict): Dictionary containing sentences for each chapter.
        tokens (dict): Dictionary containing tokenized words for each chapter.
    Returns:
        dict: Average sentence length for each chapter.
    """
    new_dict = {}
    for i in phrases.keys():
        avg_sentence_len = len(tokens[i]) / len(phrases[i])
        new_dict[i] = avg_sentence_len
    return new_dict

def no_of_punctuation(text):
    """
    Counts the number of punctuation marks in each chapter.
    Params:
        text (dict): Dictionary of text segments for each chapter.
    Returns:
        dict: Number of punctuation marks in each chapter.
    """
    no_of_ch = {}
    for i in text.keys():
        nr_p = 0
        for j in text[i]:
            if j in string.punctuation:
                nr_p += 1
        no_of_ch[i] = nr_p
    return no_of_ch

def avg_word_length(tokenized_text):
    """
    Calculates the average word length in each chapter.
    Params:
        tokenized_text (dict): Dictionary with tokenized words for each chapter.
    Returns:
        dict: Average word length for each chapter.
    """
    avg_length = {}
    for chapter, tokens in tokenized_text.items():
        total_length = sum(len(word) for word in tokens)
        avg_length[chapter] = total_length / len(tokens) if tokens else 0
    return avg_length

def lexical_diversity(tokenized_text):
    """
    Calculates the lexical diversity for each chapter (unique words / total words).
    Params:
        tokenized_text (dict): Dictionary with tokenized words for each chapter.
    Returns:
        dict: Lexical diversity for each chapter.
    """
    diversity = {}
    for chapter, tokens in tokenized_text.items():
        unique_words = set(tokens)
        diversity[chapter] = len(unique_words) / len(tokens) if tokens else 0

    return diversity

def unique_word_count(tokenized_text):
    """
    Counts the number of unique words in each chapter.
    Params:
        tokenized_text (dict): Dictionary with tokenized words for each chapter.
    Returns:
        dict: Number of unique words for each chapter.
    """
    unique_word = {}
    for chapter, tokens in tokenized_text.items():
        unique_word[chapter] = len(set(tokens))

    return unique_word

def count_syllables(word):
    """
    Counts the number of syllables in a word.
    Params:
        word (str): A single word.
    Returns:
        int: Number of syllables in the word.
    """
    word = word.lower()
    syllables = 0
    vowel_sequences = re.findall(r'[aeiouy]+', word)
    for vowel_sequence in vowel_sequences:
        syllables += 1
    if word.endswith("e"):
        syllables -= 1
    syllables = max(1, syllables)

    return syllables

def flesch_reading_ease(sentences):
    """
    Calculates the Flesch Reading Ease score for each chapter.
    Params:
        sentences (dict): Dictionary containing sentences for each chapter.
    Returns:
        dict: Flesch Reading Ease score for each chapter.
    """
    reading_ease_score = {}
    for i in sentences.keys():
        all_words = [word for sentence in sentences[i] for word in sentence.split()]
        syllables = sum(count_syllables(word) for word in all_words)

        total_sentences = len(sentences[i])
        total_words = len(all_words)
        total_syllables = syllables

        if total_sentences == 0 or total_words == 0:
            reading_ease_score[i] = 0
        else:

            # Calculate Flesch Reading Ease score using the formula
            reading_ease_score[i] = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)

    return reading_ease_score


### SENTENCE FEATURES ###

def sentence_length_by_characters(sentences):
    """
    Calculates the length of each sentence by characters.
    Params:
        sentences (dict): Dictionary containing sentences for each chapter.
    Returns:
        dict: Length of each sentence in characters.
    """
    result = {key: {s: len(s) for s in sentences[key]} for key in sentences}
    return result

def sentence_length_by_word(sentences):
    """
    Calculates the length of each sentence by words.
    Params:
        sentences (dict): Dictionary containing sentences for each chapter.
    Returns:
        dict: Length of each sentence in words.
    """
    result = {key: {s: len(s.split()) for s in sentences[key]} for key in sentences}
    return result

def sentence_avg_word_length(sentences):
    """
    Calculates the average word length in each sentence.
    Params:
        sentences (dict): Dictionary containing sentences for each chapter.
    Returns:
        dict: Average word length for each sentence.
    """
    result = {}
    for key in sentences:
        sentence_avg = {}
        for s in sentences[key]:
            words = s.split()
            sentence_avg[s] = sum(len(word) for word in words) / len(words) if words else 0
        result[key] = sentence_avg

    return result

def sentence_stopwords_count(sentences):
    """
    Counts the number of stopwords in each sentence.
    Params:
        sentences (dict): Dictionary containing sentences for each chapter.
    Returns:
        dict: Number of stopwords for each sentence.
    """
    stop_words = set(stopwords.words('english'))
    result = {}
    for key in sentences:
        sentence_stopword_count = {}
        for s in sentences[key]:
            words = s.lower().split()
            sentence_stopword_count[s] = sum(1 for word in words if word in stop_words)
        result[key] = sentence_stopword_count

    return result
