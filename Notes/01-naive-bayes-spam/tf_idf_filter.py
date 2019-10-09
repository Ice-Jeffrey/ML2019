import pandas as pd
from visualise import word_cloud
from nlp import transform
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def load_data():
    data = pd.read_csv("spam.csv", encoding="latin-1")
    data.drop(columns=data.columns[2:], inplace=True)
    data.columns = ["label", "message"]
    return data


def main():
    # 1. Load and transform
    data = load_data()
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    messages = data['message'].apply(transform)

    # 1a. TD/IDF
    messages = messages.apply(lambda x: ' '.join(x))
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(messages)
    transformer = TfidfTransformer().fit(counts)
    counts = transformer.transform(counts)

    # 2. Visualize
    #word_cloud(data[data["label"] == 0]["message"])
    #word_cloud(data[data["label"] == 1]["message"])

    # 3. Train
    X_train, X_test, y_train, y_test = train_test_split(
        counts, data['label'], test_size=0.25, random_state=69)
    c = MultinomialNB()
    c.fit(X_train, y_train)

    # 4. Test
    predictions = c.predict(X_test)
    print(accuracy_score(y_test, predictions))


if __name__ == "__main__":
    main()
