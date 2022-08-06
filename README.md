# Disaster Response Pipeline Project

### Objective:
- this is a udacity project to train skills about?
1. How to build a ETL Pipeline
2. How to build a machine learn pipeline with nlkt
    - normalize data
    - tokenize
    - remove stop words
    - train model
    - put all the steps in ml pipeline
    - evaluate results
    - use grid search to improve model predict
3. build end to end aplication to use our model in web app aplication with flask

### Files:

You will find 3 folders in this project.
1. app 
    - files to run the webapp aplication with flask 
2. data
    - csv files and process_data.py with ETL pipeline
3. models 
    - train_classifier.py to build your ml model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
