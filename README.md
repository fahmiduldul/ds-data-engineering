# Disaster Response Pipeline

## Description
In this project I tried to create ML model to predict what the needed help is from the message given from real world disaster, and deploy it to the web.


## Folder Structure
### datasets/
- messages.csv
raw file that contains all the messages used for training and testing
- categories.csv
raw file that contain all the category for each message in messages.csv
- DisasterResponse.db
sqlite database used to store cleaned data

### notebooks/
- ETL_pipeline.ipynb
Jupyter notebook file that contain ETL pipeline prototyping
- ML_Pipeline.ipynb
Jupyter notebook file that contain ML pipeline prototyping

### web-app/
contain all the files and folder to be deployed as web app
- app/backend/process_data.py
codes for ETL Pipeline based on ETL_Pipeline.ipynb file
to run this file use `python3 web-app/app/backend/process_data.py datasets/messages.csv datasets/categories.csv datasets/DisasterResponse.db`
- app/backend/train_classfier.py
codes for ML Pipeline based on MLL_Pipeline.ipynb file