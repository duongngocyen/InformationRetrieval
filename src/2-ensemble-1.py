import numpy as np
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd

test_df = pd.read_csv("../data/FINAL_TEST_DATA.csv")
test_df['sentiment'] = test_df['sentiment'].apply(lambda x: 2 if x == "positive" else (0 if x == "negative" else 1))
y_test = test_df["sentiment"].values

avg_predictions_bert = np.load("../data/average_prediction_bert.npy")
avg_predictions_roberta = np.load("../data/average_prediction_roberta.npy")
avg_predictions_distilbert = np.load("../data/average_prediction_distilbert.npy")

def report(y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
    print(classification_report(y_true, y_pred))

    # Calculate macro-average precision, recall, and F1-score
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nPrecision={macro_precision}\nRecall={macro_recall}\nF1-score={macro_f1}\nAccuracy={accuracy}")
    

initial_guess = [0, 1.25, 0.25]

def objective(coefficients):
    avg_predictions = sum(c * p for c, p in zip(coefficients, [avg_predictions_bert, avg_predictions_roberta, avg_predictions_distilbert]))
    final_predictions = np.argmax(avg_predictions, axis=1)
    return -accuracy_score(y_test, final_predictions)

result = minimize(objective, initial_guess, method='nelder-mead', tol=1e-6)

coff = result.x
print(coff)
test_pred = sum(c * p for c, p in zip(coff, [avg_predictions_bert, avg_predictions_roberta, avg_predictions_distilbert]))
test_label = np.argmax(test_pred, axis=1)
report(y_test, test_label)