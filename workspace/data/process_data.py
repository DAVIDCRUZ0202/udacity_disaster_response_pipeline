import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Reads in raw SQL data and returns a dataframe.
    
    Uses pandas to read in data from pre-determined and pre-loaded
    sources. Then combines this data on key arguments, and returns
    the merged dataset.
    
    Args:
    messages_filepath: takes the file path for where the raw
    message logs are located.
    categories_filepath: takes the file path for where the categorical
    data can be found.
    
    Returns:
    a merged dataframe of messages with categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = categories.merge(messages, on='id')
    return df

def clean_data(df):
    """
    Returns a cleaned dataframe. Cleans out duplicates,
    creates binary classes, drops constant classes.
    
    Args:
    df: The dataframe to be cleaned
    
    Returns:
    The same dataframe.
    """
    categories = df['categories'].str.split(pat=';',expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: str(x)[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    df = df.drop(['child_alone'], axis=1)
    for ind, row in df.iterrows():
        if row['related'] == 2:
            df = df.drop(ind)
    print(df.columns)
    return df


def save_data(df, database_filename):
    """
    Saves cleaned data back to a SQL database.
    
    Args:
    df: the data frame to save
    database_filename: The filepath to access where the data
    should be stored.
    
    Returns:
    Nothing but you can pull the cleaned data if you want!
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('cleandf', engine, if_exists='replace', index=False)


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
