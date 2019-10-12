import pandas as pd
from visualise import word_cloud
from learning_lib import train_test_split, accuracy_score
from nlp import transform
from bow_classifier import BagOfWordsClassifier


def load_data():
    data = pd.read_csv("Notes/01-naive-bayes-spam/spam.csv", encoding="latin-1")
    data.drop(columns=data.columns[2:], inplace=True)
    data.columns = ["label", "message"]
    return data


def main():
    # 1. Load and transform
    data = load_data()
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    data['message'] = data['message'].apply(transform)
    train_data, test_data = train_test_split(data, test_size=0.25)

    # 2. Visualize
    #word_cloud(data[data["label"] == 0]["message"])
    #word_cloud(data[data["label"] == 1]["message"])

    
    # 3. Train
    c = BagOfWordsClassifier()
    c.fit(train_data['message'], train_data['label'])

    # 4. Test
    print(c.predict(transform("This is a boring test message")))
    print(c.predict(transform("Win cash now! Text ENTER to 80975")))

    # All test data
    predictions = {}
    for (i, message) in enumerate(test_data['message']):
        predictions[i] = c.predict(message)
    print(accuracy_score(test_data['label'], predictions))
    

if __name__ == "__main__":
    main()
