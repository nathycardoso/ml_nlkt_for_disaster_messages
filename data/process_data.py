import sys
import pandas as pd
import numpy as np 

pd.set_option('display.max_colwidth', 1000)

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
        this function will load data from csv file and return one dataframe with columns from 2 csv bellow:
            messages_filepath = path to messages csv file
            categories_filepath = path to categories csv file
    
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages,categories,how='left',on='id',copy=True)

def clean_data(df):
    
    
    """
        this function will clean data from df load before
            - transform each categories in columns
            - drop duplicates
            - fill na values
            
        and return one clean dataset 
    
    """
    
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categori     # one way is to apply a lambda function that takes everng 
    # up to the second to last character of each string wslicing
    category_colnames = list(row.str.slice(stop=-2))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories.head(1):
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype('int32')

    # drop the original categories column from `df`
    del df['categories']
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    #drop null values
    df = df.fillna(0)
    
    #convert to binary
    df['related'] = df['related'].astype('str').str.replace('2', '1')
    
    return df


def save_data(df, database_filename):
    
    
    """
        this function will save the dataframe in a sqlite database
        here, you need to inform the name of the table
        in this example, we will save the data in messages table with replace argument in case the table aready exists
        
    """
    
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False,if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
