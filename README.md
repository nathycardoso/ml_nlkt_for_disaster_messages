# Disaster Response Pipeline Project

### Objective:
- this is a udacity project to identify disaster patterns in text messges. this can help people to better undestand when and where people need help, food, water and others things. Also we can train skills about:
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

app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
README.md


    
### installations:
1. To run this project you need to install/import the libs below:
    - pandas
    - numpy
    - sqlalchemy
    - re
    - nltk
    - sklearn.pipeline
    - CountVectorizer
    - TfidfTransformer
    - sklearn.datasets > make_multilabel_classification
    - sklearn.multioutput > MultiOutputClassifier
    - sklearn.ensemble > RandomForestClassifier
    - sklearn.model_selection > train_test_split
    - sklearn.metrics > classification_report

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
