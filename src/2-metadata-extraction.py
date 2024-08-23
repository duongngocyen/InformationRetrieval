
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

for data in ['FINAL_TRAIN_DATA.csv', 'FINAL_TEST_DATA.csv']:
    data['Content'] = data['Content'].fillna("NA")
    # data = train_df
    def length_score(text):
        """
        counting the length of the text, transfer to log scale and turn into score
        """
        l = len(text)
        return np.log(l) 


    def capslock_score(text):
        """
        ratio of uppercase characters to all characters in text
        +1 to remove 0, avoid trouble when transforming to log scale later
        """
        chars = ''.join(text.split())
        ratio = (sum(map(str.isupper, chars)) + 1)/ len(chars)

        return np.log(ratio)


    lemma = WordNetLemmatizer() # turn to the same form
    stop_words = stopwords.words('english')

    def text_prep(x):
        corp = str(x).lower()
        corp = re.sub('[^a-zA-Z]+',' ', corp).strip()
        tokens = word_tokenize(corp)
        words = [t for t in tokens if t not in stop_words]
        lemmatize = [lemma.lemmatize(w) for w in words]

        return lemmatize


    data['length_score'] = data['Content'].apply(length_score)
    data['capslock_score'] = data['Content'].apply(capslock_score)

    preprocessed_sentence = [text_prep(i) for i in data['Content']]
    data["preprocessed_content"] = preprocessed_sentence


    file = open('negative-words.txt', 'r', encoding="ISO-8859-1")
    neg_words = file.read().split()
    file = open('positive-words.txt', 'r', encoding="ISO-8859-1")
    pos_words = file.read().split()

    num_pos = data['preprocessed_content'].map(lambda x: np.log(len([i for i in x if i in pos_words])+2))
    data['pos_count'] = num_pos
    num_neg = data['preprocessed_content'].map(lambda x: np.log(len([i for i in x if i in neg_words])+2))
    data['neg_count'] = num_neg

    data.to_csv(f"../data/INNOVATION_{data}", index=False)