from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from numpy import zeros


def transform(sentence, lower_case=True, stem=True, stop_words=True, gram=2):
    if lower_case:
        sentence = sentence.lower()
    words = word_tokenize(sentence)
    words = [w for w in words if len(w) > 2]
    if stop_words:
        sws = stopwords.words("english")
        words = [w for w in words if w not in sws]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]
    if gram > 1:
        ws = []
        for i in range(len(words) - gram + 1):
            ws.append(' '.join(words[i:i + gram]))
        words = ws
    return words
