from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from numpy import zeros


def transform(sentence, lower_case=True, stem=True, stop_words=True, gram=1):
    #transform the sentence to lowercase
    if lower_case:
        sentence = sentence.lower()
    #tokenize the setence
    words = word_tokenize(sentence)
    #drop the characters like commas, slash
    words = [w for w in words if len(w) > 2]
    if stop_words:
        #make a list including stopwords in English
        sws = stopwords.words("english")
        #drop the stopwords
        words = [w for w in words if w not in sws]
    if stem:
        stemmer = PorterStemmer()
        #remove the words with weird spelling
        words = [stemmer.stem(w) for w in words]
    if gram > 1:
        #divide words into groups according to the gram
        ws = []
        for i in range(len(words) - gram + 1):
            ws.append(' '.join(words[i:i + gram]))
        words = ws
    return words
