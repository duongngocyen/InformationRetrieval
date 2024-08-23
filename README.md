# Classification task, Group 25

The following python files are available in the folder.

I. Data preparation files:
0-label-llm.py : file use to label training data using Chain-of-Thought
0-data-exploreation.ipynb : used for data exploration purpose.

II. Normal baseline models.
1-Vader.py : used to run vader benchmark
1-LSTM.py : used to run LSTM benchmark
1-Bert.py : used to run Bert benchmark
1-Roberta.py: used to run RoBERTa benchmark
1-Distilbert.py: used to run DistilBERT benchmark

Step to preproduce result.
1. Go in the notebook, change the file path of train dataset and test dataset. The train dataset and test dataset are also submitted in data zip file.
2. For Bert, Roberta, DistilBERT, change the number of fold and the path of model to the folder you want to save the model to.
3. Run file by: python <filename>

III. Enhancement models
2-ensemble-1.py : used for ensemble learning in 3 models.
2-metadata-extraction.py : used to extract data for multimodal learning. This file need to run first before multimodal-learning.py. However, we also publish the data in data folder, start with INNOVATION*, indicate our dataset with extracted metadata, and can be used straightly in multimodal-transformers.py.
2-multimodal-learning.py : notebook used to train multimodal transformers for text and tabular data.
2-ensemble-2.py : notebook used to combine 4 models (BERT, RoBERTa, DistilBERT, multimodal model)

Enhancement 1: Ensemble learning:
Go in 2-ensemble-1.py, change the data paths for training, testing, and features. We upload all data in data zip folder. For features, we note the file with FEATURES* start.

Enhancement 2: Multimodal learning:
Step 1: Run 2-metadata-extraction.py notebook to extract metadata.
Step 2: Use the output file in step 1, to pass to notebook 2-multimodal-transformers.py and run by python <filename>.

Enhancement 3: Ensemble learning + multimodal learning
Go in 2-ensemble-2.py, change the data paths for training, testing, and features. We upload all data in data zip folder. For features, we note the file with FEATURES* start.