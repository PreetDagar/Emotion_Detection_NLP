import joblib
import string
from nltk.corpus import stopwords

vectorizer = joblib.load('bow_vectorizer.pkl')
model = joblib.load('logistic_regression_model.pkl')

stop_word = set(stopwords.words('english'))

stop_word.difference_update(['not', 'no', 'never', 'nor'])

def str_rem(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(txt):
    new = ''

    for i in txt:
        if not i.isdigit():
            new += i

    return new

def remove_stopwords(txt):
    words = txt.split()
    cleaned = []

    for word in words:
        if word not in stop_word:
            cleaned.append(word)

    return ' '.join(cleaned)

def preprocess_text(txt):
    txt = txt.lower()
    txt = str_rem(txt)
    txt = remove_numbers(txt)
    txt = remove_stopwords(txt)

    return txt

reverse_mapping = {
    0: 'sadness',
    1: 'anger',
    2: 'love',
    3: 'surprise',
    4: 'fear',
    5: 'joy'
}


def predict_emotion(sentence):

    cleaned_sentence = preprocess_text(sentence)

    vector = vectorizer.transform([cleaned_sentence])

    prediction = model.predict(vector)[0]

    emotion = reverse_mapping[prediction]

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities) * 100

    print("Confidence:", round(confidence, 2), "%")

    return emotion

if __name__ == "__main__":

    sentence = input("Enter a sentence: ")

    result = predict_emotion(sentence)

    print("Predicted emotion:", result)