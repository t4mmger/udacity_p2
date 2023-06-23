import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """
    load_data does: 
    - loading the data,
    - joining messages and categories, 
    - splitting categories, so each category is in one column,
    - extracting column names and changing the values into dummies,
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how = 'inner', on = 'id')
    categories36 = categories['categories'].str.split(pat = ';', expand = True)
    row = categories36.iloc[0]
    category_colnames = 'cat_' + row.apply(lambda x: x[:-2])
    categories36.columns = category_colnames

    for column in categories36:
        # set each value to be the last character of the string
        categories36[column] = categories36[column].astype(str).str.slice(-1)
        
        # convert column from string to numeric
        categories36[column] = pd.to_numeric(categories36[column])

    df.drop(columns = 'categories', inplace = True)
    df = pd.concat([df, categories36], axis=1)

    return df


def clean_data(df):
    """
    clean_data does: 
     - removing duplicates caused by non-unique id column.
     - removing rows that have values > 1 in responsce columns,
     - removing response columns that have no variety (e.g. all values are 0).
     - removing rows that have NA in response columns. 
    """
    df.drop_duplicates(inplace = True)

    for column in df.filter(regex='^cat_').columns:
        if df[column].max() > 1:
            print(df[column].value_counts())
            print('records > 1 will be deleted.')
            df.drop(df[df[column] > 1].index, inplace=True)

        if df[column].max() == df[column].min():
            df.drop(column, axis=1, inplace = True) 
            print('column', column, 'dropped because no variety in data.')

    df.dropna(inplace=True, subset = df.filter(regex='^cat_').columns) 

    return df


def save_data(df, database_filename):
    """
    save_data does:
    - saving the df as a sql db under the path provided by the user
    """
    location = 'sqlite:///{}'.format(database_filename)
    #I got the idea for location from https://stackoverflow.com/questions/3247183/variable-table-name-in-sqlite
    engine = create_engine(location)
    df.to_sql('DisasterResponse', engine, if_exists = 'replace', index=False)
    pass  

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