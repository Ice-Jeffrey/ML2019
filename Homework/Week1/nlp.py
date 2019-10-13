from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from numpy import zeros


def transform(figures, lower_case=True, stem=True, stop_words=True, gram=1):
    words = word_tokenize(figures)
    if gram > 1:
        #divide words into groups according to the gram
        ws = []
        for i in range(len(figures) - gram + 1):
            ws.append(' '.join(figures[i:i + gram]))
        words = ws
    return words