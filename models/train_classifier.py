import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    
    
    """
        call this function to load the data from database
        you this need to pass the database filepath
        this function will return 3 variables with:
            - X = flat messages array to use in your model
            - Y = the flag 1 or 0 for each label
            - category_names = a list of label names 
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df[['message']].values
    X = flat_list = [item for sublist in X.tolist() for item in sublist]
    Y = df.drop(['id','message','original','genre'], axis=1).values
    category_names = df.drop(['id','message','original','genre'], axis=1).columns
    
    return X, Y, category_names

def tokenize(text):
    
    """
    
        this function will ajust the text to use in ml pipeline
            - normalize removing some caracters
            - tokenize words with nlkt lib
            - remove stop words
            - lemmatizer each word
            - stemmed each word
            
        and finaly return a clean array of words
    
    """
    
    text_normalized = re.sub(r"[^a-zA-Z0-9]", " ", str(text).lower())
    word_tokens = word_tokenize(text_normalized)
    no_stop_words = [w for w in word_tokens if w not in stopwords.words('english')]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in no_stop_words]
    stemmed = [PorterStemmer().stem(w) for w in lemmed]
    
    return stemmed


def build_model():
    """
        this function is to build ml pipeline with all steps to improve your model
        they return a pipeline object to train and predict in other steps
    
    """
    pipeline = Pipeline([
        ('tokenize', CountVectorizer(tokenizer=tokenize)),
        ('tfid', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    
    """
        this function will evaluate the model and print the f1-score for each feature
    
    """
    y_pred = model.predict(X_test)
    print(classification_report(np.hstack(Y_test), np.hstack(y_pred), target_names=category_names))


def save_model(model, model_filepath):
    
    """
        this function will save your model to use in web app aplication to predict new messages
    """
    
    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()