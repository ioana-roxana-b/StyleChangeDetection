import string
from nltk.corpus import stopwords
import re

def no_of_words(tokenized_text):
    no_of_w = {chapter: len(tokens) for chapter, tokens in tokenized_text.items()}
    return no_of_w

def no_of_stop_words(tokenized_text, tokenized_text_sw):
    no_of_sw = {}
    for (i, j) in zip(tokenized_text.keys(), tokenized_text_sw.keys()):
        no_of_sw[i] = len(tokenized_text[i]) - len(tokenized_text_sw[j])
    return no_of_sw

def no_of_contracted_wordforms(tokenized_text):
    pattern = r"\b\w+'\w+\b"
    for i in tokenized_text.keys():
        text = ' '.join(tokenized_text[i])
        contracted_word_forms = re.findall(pattern, text)
        num_contracted_word_forms = len(contracted_word_forms)
        tokenized_text[i] = num_contracted_word_forms
    return tokenized_text

def no_of_characters(text):
    no_of_ch = {}
    for i in text.keys():
        no_of_ch[i] = len(text[i])
    return no_of_ch

def no_of_sentences(phrases):
    no_of_s = {}
    for i in phrases.keys():
        no_of_s[i] = len(phrases[i])
    return no_of_s

def avg_sentence_length(phrases, no_words):
    for i in phrases.keys():
        avg_sentence_len = len(no_words[i]) / len(phrases[i])
        phrases[i] = avg_sentence_len
    return phrases

def no_of_punctuation(text):
    no_of_ch = {}
    for i in text.keys():
        nr_p = 0
        for j in text[i]:
            if j in string.punctuation:
                nr_p +=1
        no_of_ch[i] = nr_p
    return no_of_ch

def avg_word_length(tokenized_text):
    avg_length = {}
    for chapter, tokens in tokenized_text.items():
        total_length = sum(len(word) for word in tokens)
        if len(tokens) > 0:
            avg_length[chapter] = total_length / len(tokens)
        else:
            avg_length[chapter] = 0
    return avg_length

def lexical_diversity(tokenized_text):
    diversity = {}
    for chapter, tokens in tokenized_text.items():
        unique_words = set(tokens)
        diversity[chapter] = len(unique_words) / len(tokens) if tokens else 0
    return diversity

### SENTENCE FEATURES ###
def sentence_length_by_characters(sentences):
    result = {key: {s: len(s) for s in sentences[key]} for key in sentences}
    return result

def sentence_length_by_word(sentences):
    result = {key: {s: len(s.split()) for s in sentences[key]} for key in sentences}
    return result

def sentence_avg_word_length(sentences):
    result = {}
    for key in sentences:
        sentence_avg = {}
        for s in sentences[key]:
            words = s.split()
            if words:  # Check if the list is not empty
                sentence_avg[s] = sum(len(word) for word in words) / len(words)
            else:
                sentence_avg[s] = 0
        result[key] = sentence_avg
    return result

def sentence_stopwords_count(sentences):
    stop_words = set(stopwords.words('english'))
    result = {}
    for key in sentences:
        sentence_stopword_count = {}
        for s in sentences[key]:
            words = s.lower().split()
            sentence_stopword_count[s] = sum(1 for word in words if word in stop_words)
        result[key] = sentence_stopword_count
    return result

