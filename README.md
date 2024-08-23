# Classification of Post

This repository contains scripts and notebooks for classifying posts using various machine learning models. The files are organized into sections based on their purpose, including data preparation, baseline models, and enhanced models.

## I. Data Preparation Files
- **`0-label-llm.py`**: Labels training data using Chain-of-Thought.
- **`0-data-exploration.ipynb`**: Used for data exploration purposes.

## II. Baseline Models
- **`1-Vader.py`**: Runs the VADER sentiment analysis benchmark.
- **`1-LSTM.py`**: Runs the LSTM benchmark.
- **`1-Bert.py`**: Runs the BERT benchmark.
- **`1-Roberta.py`**: Runs the RoBERTa benchmark.
- **`1-Distilbert.py`**: Runs the DistilBERT benchmark.

### Steps to Reproduce Results
1. Open the relevant notebook and update the file paths for the training and testing datasets. The datasets are included in the provided data ZIP file.
2. For BERT, RoBERTa, and DistilBERT, modify the number of folds and the model save path as needed.
3. Run the scripts using the command: `python <filename>`.

## III. Enhancement Models
- **`2-ensemble-1.py`**: Implements ensemble learning across three models.
- **`2-metadata-extraction.py`**: Extracts data for multimodal learning. This script must be run before `2-multimodal-learning.py`. Pre-extracted metadata is also provided in the data folder (files starting with `INNOVATION*`), and these can be directly used in `2-multimodal-transformers.py`.
- **`2-multimodal-learning.py`**: Notebook for training multimodal transformers on text and tabular data.
- **`2-ensemble-2.py`**: Combines four models (BERT, RoBERTa, DistilBERT, and the multimodal model) using ensemble learning.

### Enhancement 1: Ensemble Learning
1. Open `2-ensemble-1.py` and update the paths for training data, testing data, and feature files. All necessary data is included in the provided data ZIP file. Feature files are marked with `FEATURES*`.

### Enhancement 2: Multimodal Learning
1. Run the `2-metadata-extraction.py` notebook to extract metadata.
2. Use the output from Step 1 as input for the `2-multimodal-learning.py` notebook and run it using the command: `python <filename>`.

### Enhancement 3: Ensemble Learning + Multimodal Learning
1. Open `2-ensemble-2.py` and update the paths for training data, testing data, and feature files. All necessary data is included in the provided data ZIP file. Feature files are marked with `FEATURES*`.
