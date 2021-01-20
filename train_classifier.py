import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

import nltk
nltk.download(['punkt', 'wordnet'])

import pickle

def load_data(database_filepath):
    '''
    Loads data from sql database
        
    Params
    ======
        database_filepath (str): filepath of database
    Returns
    ======
        X (pandas series): messages
        Y (pandas dataframe): classifications of messages
        category names (list): names of categories
    '''
   
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('df', con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    
    category_names = list(Y)

    return X, Y, category_names

def tokenize(text):
    '''
    Cleans messages for model 
        
    Params
    ======
        text (str): text to process
    Returns
    ======
        clean_tokens (list): cleaned text

    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    Build data pipeline for classification of messages
        
    Returns
    ======
        cv (sklearn model): model pipeline

    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
    'clf__estimator__n_estimators': [10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates model on test data. Prints report for each category.
        
    Params
    ======
        model: model
        X_test (pandas series): input messages to be classified
        Y_test (pandas dataframe): classifications of test data
        category names (list): names of categories

    '''
    y_pred = model.predict(X_test)

    for i in range(y_pred.shape[1]):
        print(classification_report(Y_test.iloc[:,i].values, y_pred[:,i]))

def save_model(model, model_filepath):
    '''
    Saves model as pkl file.
        
    Params
    ======
        model (sklearn model): model
        model_filepath (str): filepath to save model
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    '''
    Trains classifier of disaster response messages stored in an sql database.
    '''
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