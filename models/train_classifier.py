# import libraries

import sys
import nltk 
nltk.download(['omw-1.4', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine, inspect

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

def load_data(database_filepath):
    """
    load_data does:
    - import the table in DisasterResponse-DB,
    - Creates X, y for the regressions,
    - Creates category_names. 
    """

    location = 'sqlite:///{}'.format(database_filepath)
    engine = create_engine(location)
    #engine.table_names()
    Inspector = inspect(engine)
    Inspector.get_table_names()

    df = pd.read_sql_table('DisasterResponse', engine)
    #X = df[['message','genre']]
    X = df['message']
    y = df.filter(regex='^cat_').values
    category_names = df.filter(regex='^cat_').columns
    return X, y, category_names
    


def tokenize(text):
    """
    tokenize does:
    - split the test into words,
    - lemmentize the words,
    - makes word lowercase and strips unneccesary space
    """
    #From the Udacity course
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    build_model does: 
    - Create a model with a pipeline to make predictions,
    - Using grid search to find the best parameters,
    - max-iter was increased from 100 (default) to 200 to get better convergence.
    """

    pipeline = Pipeline([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),

                ('clf', MultiOutputClassifier(LogisticRegression(C=1, penalty='l1', solver='liblinear', max_iter = 200)))
                ])

    parameters = {'clf__estimator__penalty': ['l1', 'l2']}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    model = cv
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate_model does:
    - calculate the precision, recall, f1-score and support for each category.
    - 
    """

    y_pred = model.predict(X_test)
    df_report_results = pd.DataFrame() #create empty df to fill the report results
    for i, c in enumerate(category_names):
        
        report = classification_report(y_true = Y_test[:,i], y_pred = y_pred[:,i], output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report['cateory'] = c
        df_report_results = pd.concat([df_report_results, df_report])

    df_report_results.to_csv('classification_report.csv', index=True)

    pass


def save_model(model, model_filepath):
    """
    save_model does:
    - saves the model under model_filepath
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

    pass


def main():
    """
    main does:
    - run the functions load_data(), build_model(), evaluate_model(), save_model()
    - it also fits the model.  
    """
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