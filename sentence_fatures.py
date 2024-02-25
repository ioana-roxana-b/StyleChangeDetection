import dataset
from nltk.corpus import stopwords

def sentence_length_by_characters(sentence):
    lengths = {}
    for i in sentence.keys():
        for j in sentence[i]:
            lengths[j]=len(j)
        sentence[i] = lengths
        lengths = {}

    return sentence

def sentence_length_by_word(sentence):
    lengths = {}
    for i in sentence.keys():
        for j in sentence[i]:
            words = j.split()
            lengths[j] = len(words)
        sentence[i] = lengths
        lengths = {}
    return sentence

def avg_word_length(sentence):
    lengths = {}
    for i in sentence.keys():
        for j in sentence[i]:
            words = j.split()
            if len(words):
                avg = sum(len(word) for word in words) / len(words)
                lengths[j] = avg
        sentence[i] = lengths
        lengths = {}
    return sentence

def stopwords_count(sentence):
    stop_words = set(stopwords.words('english'))
    stopword_count = {}
    for i in sentence.keys():
        for j in sentence[i]:
            sentence = str.lower(j)
            words = sentence.split()
            stopword_count[j] = sum([1 for word in words if word in stop_words])
        sentence[i] = stopword_count
        stopword_count = {}
    return sentence

