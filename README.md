# Disaster Response Pipeline

## Description

In this project I tried to create ML model to predict what the needed help is from the message given from real world disaster, and deploy it to the web.

## Folder Structure

### datasets/

- `messages.csv`
raw file that contains all the messages used for training and testing
- `categories.csv`
raw file that contain all the category for each message in messages.csv
- `DisasterResponse.db`
sqlite database used to store cleaned data
- `datasets/process_data.py`
codes for ETL Pipeline based on ETL_Pipeline.ipynb file
to run this file use `python3 datasets/process_data.py datasets/messages.csv datasets/categories.csv datasets/DisasterResponse.db` from root folder

### notebooks/

- `ETL_pipeline.ipynb`
Jupyter notebook file that contain ETL pipeline prototyping
- `ML_Pipeline.ipynb`
Jupyter notebook file that contain ML pipeline prototyping

### web-app/

contain all the files and folder to be deployed as web app

- `app/train_classfier.py`
codes for ML Pipeline based on ML_Pipeline.ipynb file
to run this file use `python3 train_classifier.py database/DisasterResponse.db models/test_model.pickle test`. "test" argument used for testing with only 100 rows
- `tokenizer.py`
contain tokenize function so the model can be loaded to `run.py` file
