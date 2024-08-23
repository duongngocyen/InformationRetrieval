import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

import torch
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import re

train_path = '../data/FINAL_TRAIN_DATA.csv'
test_path = '../data/FINAL_TEST_DATA.csv'

# If we have a GPU available, we'll set our device to GPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    
train_df = pd.read_csv(train_path)
train_df.drop_duplicates(inplace=True)
train_df.reset_index(drop=True, inplace=True)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

preprocessed_content = []
for review in train_df['Content']:
    preprocessed_content.append(preprocess_text(review))

def get_all_tokens():
    all_tokens = set()
    i = 0
    for review in preprocessed_content:
        data = set(review.split())
        all_tokens.update(data)
    return all_tokens

all_tokens = get_all_tokens()
# dictionary where key is the token and index is the value
# this is for look up
def build_vocab(all_tokens):
    vocab = {}
    vocab['PAD'] = 0
    vocab['UNK'] = 1 # handle words that are not seen before
    i = 2
    for val in all_tokens:
        vocab[val] = i
        i += 1
    return vocab

vocab = build_vocab(all_tokens)
print(len(vocab))

import itertools

for i, val in enumerate(itertools.islice(vocab, 10)):
    print(i,val)
    
    
####### ENCODE REVIEW ########
def encode_review(text):
    indexes = []

    r = preprocess_text(text)
    for token in r.split():
        try:
            indexes.append(vocab[token])
        # print(token)
        except KeyError:
        #  print(token)
            indexes.append(vocab['UNK'])

    return indexes

train_df['Encoded_Review'] = train_df['Content'].apply(encode_review)

# CHECK LENGTH
lengths = [len(review) for review in train_df['Encoded_Review']]
print(len(lengths))


def truncate_or_padding(max_len, review):
    if len(review) > max_len:
        review = review[:max_len]
    else:
        review = review + [0] * (max_len - len(review))
    return review

max_len=500
train_df['Encoded_Review'] = train_df['Encoded_Review'].apply(lambda x: truncate_or_padding(max_len, x))

lengths = [len(review) for review in train_df['Encoded_Review']]

train_df['sentiment_1'] = train_df['sentiment'].apply(lambda x: 1 if x in ['positive', 'negative'] else 0)
# train-test split for task 1
X_train, X_val, y_train, y_val = train_test_split(train_df['Encoded_Review'], train_df['sentiment_1'], test_size=0.1, random_state=1)

print(len(X_train))
print(len(y_train))
print(len(X_val))
print(len(y_val))

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.tolist(), dtype=torch.int64)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]


def intialise_loader(X, y, batch):
    dataset = CustomDataset(X, y)
    d_loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    return d_loader


batch_size = 512
train_dataloader = intialise_loader(X_train, y_train, batch_size)
val_dataloader = intialise_loader(X_val, y_val, batch_size)

# parameters
embedding_dim = 300
hidden_dim = 200
output_dim = 2
seed = 10
no_layers=1

class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()

        # embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=no_layers, batch_first=True)

        # linear layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, X_batch, pool):
        h0 = torch.zeros((no_layers,X_batch.shape[0],hidden_dim)).to(device)
        c0 = torch.zeros((no_layers,X_batch.shape[0],hidden_dim)).to(device)
        hidden = (h0,c0)
        embeddings = self.embedding_layer(X_batch)
        out,_ = self.lstm(embeddings,hidden)
        if pool == True:
            out = torch.mean(out, dim=1)
            output = self.linear(out)
        else:
            output = self.linear(out[:, -1, :])  # Consider only the output of the last time step
        return output
    
    
epochs = 20
lr = 0.001

loss_fn = nn.CrossEntropyLoss()

def train_loop(dataloader, model, loss_fn, optimizer, pool):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss

        X, y = X.to(device), y.to(device)
        pred = model(X, pool)
        loss = loss_fn(pred, y)

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        #correct = (pred.view(-1, 2).argmax(1) == y.view(-1)).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(correct*100):>0.1f}%, Avg loss: {train_loss:>8f} \n")

    return train_loss, correct

from sklearn.metrics import f1_score

def test_loop(dataloader, model, loss_fn, pool):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct, f1 = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X,pool)

            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            pred_argmax = torch.argmax(pred, dim=1)
            f1 += f1_score(y.cpu(), pred_argmax.cpu())

    correct /= size
    val_loss /= num_batches
    f1 /= num_batches
    print(f"Val Error: \n Accuracy: {(100*correct):>0.1f}%, f1 score: {f1} \n")
    return val_loss, correct, f1

# no pooling results

model_1 = LSTMClassifier().to(device)
optimizer = Adam(model_1.parameters(), lr=lr)

train_loss, test_loss = [], []
train_acc, test_acc = [], []
f1_scores = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss, acc = train_loop(train_dataloader, model_1, loss_fn, optimizer, False)
    train_loss.append(loss), train_acc.append(acc)

    loss, acc, f1 = test_loop(val_dataloader, model_1, loss_fn, False)
    test_loss.append(loss), test_acc.append(acc), f1_scores.append(f1)

print("Done!")


# pooling results

model_2 = LSTMClassifier().to(device)
optimizer = Adam(model_2.parameters(), lr=lr)

train_loss2, test_loss2 = [], []
train_acc2, test_acc2 = [], []
f1_score2 = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss, acc = train_loop(train_dataloader, model_2, loss_fn, optimizer, True)
    train_loss2.append(loss), train_acc2.append(acc)

    loss, acc, f1 = test_loop(val_dataloader, model_2, loss_fn, True)
    test_loss2.append(loss), test_acc2.append(acc), f1_score2.append(f1)

print("Done!")

class BiLSTMClassifier(nn.Module):
    def __init__(self):
        super(BiLSTMClassifier, self).__init__()

        # embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=no_layers, batch_first=True,
                            bidirectional=True)

        # linear layer
        self.linear = nn.Linear(hidden_dim*2, 2)

    def forward(self, X_batch, pool):
        h0 = torch.zeros((no_layers*2,X_batch.shape[0],hidden_dim)).to(device)
        c0 = torch.zeros((no_layers*2,X_batch.shape[0],hidden_dim)).to(device)
        hidden = (h0,c0)
        embeddings = self.embedding_layer(X_batch)
        out,_ = self.lstm(embeddings,hidden)
        if pool == True:
            out = torch.mean(out, dim=1)
            output = self.linear(out)
        else:
            output = self.linear(out[:, -1, :])  # Consider only the output of the last time step
        return output
    
    
loss_fn = nn.CrossEntropyLoss()
model_3 = BiLSTMClassifier().to(device)
optimizer = Adam(model_3.parameters(), lr=lr)
model_3

# bilstm results

train_loss3, test_loss3 = [], []
train_acc3, test_acc3 = [], []
f1_score3 = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss, acc = train_loop(train_dataloader, model_3, loss_fn, optimizer, True)
    train_loss3.append(loss), train_acc3.append(acc)

    loss, acc, f1 = test_loop(val_dataloader, model_3, loss_fn, True)
    test_loss3.append(loss), test_acc3.append(acc), f1_score3.append(f1)

print("Done!")


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(range(epochs),test_acc, label='LSTM')
plt.plot(range(epochs),test_acc2,label='LSTM with pooling')
plt.plot(range(epochs),test_acc3,label='Bi-LSTM with pooling')

plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.save("LSTM_task1.png")
plt.legend()



################# TASK 2 ###########
# running the exact same code for task 2

train_df = train_df[train_df['sentiment']!='neutral'].reset_index(drop=True)
train_df['sentiment_2'] = train_df['sentiment'].apply(lambda x: 1 if x in ["positive"] else 0)

X_train, X_val, y_train, y_val = train_test_split(train_df['Encoded_Review'], train_df['sentiment_2'], test_size=0.1, random_state=1)

batch_size = 512
train_dataloader = intialise_loader(X_train, y_train, batch_size)
val_dataloader = intialise_loader(X_val, y_val, batch_size)

# no pooling results

model_4 = LSTMClassifier().to(device)
optimizer = Adam(model_4.parameters(), lr=lr)

train_loss, test_loss = [], []
train_acc, test_acc = [], []
f1_scores = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss, acc = train_loop(train_dataloader, model_4, loss_fn, optimizer, False)
    train_loss.append(loss), train_acc.append(acc)

    loss, acc, f1 = test_loop(val_dataloader, model_4, loss_fn, False)
    test_loss.append(loss), test_acc.append(acc), f1_scores.append(f1)

print("Done!")


# pooling results
model_5 = LSTMClassifier().to(device)
optimizer = Adam(model_5.parameters(), lr=lr)

train_loss2, test_loss2 = [], []
train_acc2, test_acc2 = [], []
f1_score2 = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss, acc = train_loop(train_dataloader, model_5, loss_fn, optimizer, True)
    train_loss2.append(loss), train_acc2.append(acc)

    loss, acc, f1 = test_loop(val_dataloader, model_5, loss_fn, True)
    test_loss2.append(loss), test_acc2.append(acc), f1_score2.append(f1)

print("Done!")


loss_fn = nn.CrossEntropyLoss()
model_6 = BiLSTMClassifier().to(device)
optimizer = Adam(model_6.parameters(), lr=lr)
model_6

# bilstm results

train_loss3, test_loss3 = [], []
train_acc3, test_acc3 = [], []
f1_score3 = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss, acc = train_loop(train_dataloader, model_6, loss_fn, optimizer, True)
    train_loss3.append(loss), train_acc3.append(acc)

    loss, acc, f1 = test_loop(val_dataloader, model_6, loss_fn, True)
    test_loss3.append(loss), test_acc3.append(acc), f1_score3.append(f1)

print("Done!")

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(range(epochs),test_acc, label='LSTM')
plt.plot(range(epochs),test_acc2,label='LSTM with pooling')
plt.plot(range(epochs),test_acc3,label='Bi-LSTM with pooling')

plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.save("LSTM_task2.png")
plt.legend()

######### COMBINING TASK 1 AND 2 ########
test_df = pd.read_csv(test_path)
# Predict opinion/non opinion first
test_df['sentiment_1'] = test_df['sentiment'].apply(lambda x: 1 if x in ['positive', 'negative'] else 0)
test_df['Encoded_Review'] = test_df['Content'].apply(encode_review)

max_len=500
test_df['Encoded_Review'] = test_df['Encoded_Review'].apply(lambda x: truncate_or_padding(max_len, x))

X = test_df['Encoded_Review']
y = test_df['sentiment_1']

batch_size = test_df.shape[0]

test_dataset = CustomDataset(X, y)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

def serve_loop(dataloader, model, loss_fn, pool):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, f1 = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X,pool)

            test_loss += loss_fn(pred, y).item()
            pred_argmax = torch.argmax(pred, dim=1)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            f1 += f1_score(y.cpu(), pred_argmax.cpu(), average='macro')
            # print(pred_argmax)

    correct /= size
    test_loss /= num_batches
    f1 /= num_batches
    print(f"Val Error: \n Accuracy: {(100*correct):>0.1f}%, f1 score: {f1} \n")
    return test_loss, correct, f1, pred_argmax.cpu().numpy()


test_loss, correct, f1, prd1 = serve_loop(test_dataloader, model_3, loss_fn, True)  # using bi-lstm
test_df['Predicted_Sentiment_1'] = prd1
# Correctly classified neutrals
correct_neutral = ((test_df['Predicted_Sentiment_1'] == 0) & (test_df['sentiment_1'] == 0)).sum()
# Predict positive/negative


#### PREDICT 2
test_df_2 = test_df[test_df['sentiment'] != 'neutral']
test_df_2['sentiment_2'] = test_df_2['sentiment'].apply(lambda x: 1 if x in ["positive"] else 0)
X = test_df_2['Encoded_Review']
y = test_df_2['sentiment_2']
batch_size = test_df_2.shape[0]
test_dataloader = intialise_loader(X, y, batch_size)
test_loss, correct, f1, prd2 = serve_loop(test_dataloader, model_6, loss_fn, True)  # using bi-lstm
test_df_2['Predicted_Sentiment_2'] = prd2
correct_opinionated = (test_df_2['Predicted_Sentiment_2'] == test_df_2['sentiment_2']).sum()

# final accuracy
accuracy = (correct_opinionated + correct_neutral)/test_df.shape[0]

# precision and recall

TP_1 = ((test_df['Predicted_Sentiment_1'] == 1) & (test_df['sentiment_1'] == 1)).sum()
TP_2 = ((test_df_2['Predicted_Sentiment_2'] == 1) & (test_df_2['sentiment_2'] == 1)).sum()
TP = TP_1 + TP_2

FP_1 = ((test_df['Predicted_Sentiment_1'] == 1) & (test_df['sentiment_1'] == 0)).sum()
FP_2 = ((test_df_2['Predicted_Sentiment_2'] == 1) & (test_df_2['sentiment_2'] == 0)).sum()
FP = FP_1 + FP_2

FN_1 = ((test_df['Predicted_Sentiment_1'] == 0) & (test_df['sentiment_1'] == 1)).sum()
FN_2 = ((test_df_2['Predicted_Sentiment_2'] == 0) & (test_df_2['sentiment_2'] == 1)).sum()
FN = FN_1 + FN_2

precision = TP / (TP + FP) if TP + FP != 0 else 0

recall = TP / (TP + FN) if TP + FN != 0 else 0

f1_score = (2*precision*recall)/(precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("f1-score:", f1_score)


