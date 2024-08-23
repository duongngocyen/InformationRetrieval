import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

train = pd.read_csv('../data/FINAL_TRAIN_DATA.csv')
test = pd.read_csv("../data/FINAL_TEST_DATA.csv")
train['sentiment'] = train['sentiment'].apply(lambda x: 2 if x == "positive" else (0 if x == "negative" else 1))
test['sentiment'] = test['sentiment'].apply(lambda x: 2 if x == "positive" else (0 if x == "negative" else 1))


lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove short words
    tokens = [token for token in tokens if len(token) > 2]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text


def report(y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
    print(classification_report(y_true, y_pred))
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nPrecision={macro_precision}\nRecall={macro_recall}\nF1-score={macro_f1}\nAccuracy={accuracy}")
    
    
X_train = train['Content'].apply(preprocess_text)
y_train = train['sentiment']

X_test = test['Content'].apply(preprocess_text)
y_test = test['sentiment']

def classify_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        return 2  # Positive
    elif sentiment_scores['compound'] <= -0.05:
        return 0  # Negative
    else:
        return 1  # Neutral

test['vader'] = X_test.apply(classify_sentiment)
report(test['vader'], test['sentiment'])