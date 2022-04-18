import sys
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
import pandas as pd
import numpy as np 
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib

def load_data(database_filepath):
    """
    Loads data from a SQL source and converts data into pandas dataframe format.
    
    Args:
    database_filepath: The file path for the raw sql database to access.
    
    Returns:
    a Feature array, Target array, and target labels list from 
    a database.
    """
    engine = create_engine('sqlite:///cleandf.db')
    df = pd.read_sql("SELECT * FROM cleandf", engine)
    X = df[['message', 'original', 'genre']]
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    cols = [col for col in Y.columns]
    return X, Y, cols

def tokenize(text):
    """
    Pre-processing function for text data.
    
    Args:
    text: Takes body of text (string format)
    
    Returns:
    Cleaned text data which has been normalized, tokenized, and lemmatized. Also 
    stop words are removed.
    """
    tokens = word_tokenize(text)
    lemma = WordNetLemmatizer()
    clean_toks = []
    for tok in tokens:
        clean_tok = lemma.lemmatize(tok).lower().strip()
        clean_toks.append(clean_tok)
    clean_toks = [word for word in clean_toks if word not in stopwords.words("english")]
    return clean_toks

def build_model():
    """
    Build an NLP ready to be trained on data.
    
    Args:
    None
    
    Returns:
    a pipeline with pre-defined hyperparameters. To save this, assign this function
    call to an arbitrary variable.
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))])
    
    parameters = {
    'clf__estimator__n_estimators':[200],
    'clf__estimator__max_features':["auto", "log2"],
    'clf__estimator__class_weight':["balanced"]}

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=-1, cv=2, verbose=3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates a model performance against test data.
    
    Args:
    model: The NLP model to be evaluated
    X_test: Feature array to be passed into our model
    Y_test: Target array to be evaluated against
    category_names: Array of target names for clean printing of
    evaluation metrics
    
    Returns:
    An evaluation of the passed in model utilizing SKLearn's
    'classification_report' function.
    """
    preds = model.predict(X_test['message'])
    for i, col in enumerate(category_names):
        print(classification_report(Y_test.iloc[:,i], preds[:,i]))

def save_model(model, model_filepath):
    """
    Saves preferred model to local file system. 
    
    Uses joblib to compress and store the model at a given location.
    
    Args:
    model: model to save
    model_filepath: desired file location for where the model will go
    
    Returns:
    Nothing but your model should be found where you sent it !
    """
    joblib.dump(model, model_filepath)
                                    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train['message'], Y_train)
        
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
