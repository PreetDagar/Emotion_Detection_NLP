# Emotion Detection NLP

An NLP-based machine learning application that predicts the emotion expressed in a given English sentence.

The project uses Natural Language Processing techniques and machine learning classification algorithms to identify six emotions: Joy, Sadness, Anger, Fear, Love, and Surprise.

## Features

- Text preprocessing and cleaning
- Bag of Words (BoW) feature extraction
- TF-IDF feature extraction
- Logistic Regression classification
- Naive Bayes classification
- Emotion prediction for custom text
- Prediction confidence scores
- Interactive Streamlit web application

## Emotions Classified

- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

## Machine Learning Models

The project experiments with two classification algorithms:

1. Logistic Regression
2. Multinomial Naive Bayes

Feature extraction methods:

- Bag of Words
- TF-IDF

## Model Performance

| Model | Feature Extraction | Accuracy |
|---|---|---:|
| Logistic Regression | Bag of Words | 87.22% |
| Naive Bayes | Bag of Words | 74.49% |
| Logistic Regression | TF-IDF | 82.15% |
| Naive Bayes | TF-IDF | 64.20% |

*Results are based on the model evaluation performed in the training notebook.*

## Project Structure

```text
Emotion_Detection_NLP/
├── app.py
├── predict.py
├── bow_vectorizer.pkl
├── logistic_regression_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/PreetDagar/Emotion_Detection_NLP.git
cd Emotion_Detection_NLP
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Download NLTK stopwords:

```bash
python3 -c "import nltk; nltk.download('stopwords')"
```

## Run the Application

Start the Streamlit application:

```bash
python3 -m streamlit run app.py
```

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Streamlit
- Joblib

## Future Improvements

- Improve classification performance for minority emotion classes.
- Experiment with advanced NLP techniques.
- Add support for more languages.
- Deploy the application online.

## Author

Preet Dagar

GitHub: https://github.com/PreetDagar
