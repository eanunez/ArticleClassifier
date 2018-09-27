"""
Description: Given the existing Sklearn model/classifier, the script predicts number of documents.

"""

from sklearn.externals import joblib
import pandas as pd
import argparse
import os


def predict_bulk(vectorizer_fn, model_fn, df):
    """Predicts number of documents.

    :param vectorizer_fn: file name of the vectorizer
    :param model_fn: file name of the model
    :param df: dataframe with columns 'input' or list of contents to predict.

    :return result: data frame of columns, 'input' and 'predicted'."""
    vect = joblib.load(vectorizer_fn)
    model = joblib.load(model_fn)

    if isinstance(df, pd.DataFrame):
        df['predicted'] = [model.predict(vect.transform(content))[0] for content in df['input']]
        return df
    elif isinstance(df, list):
        result = pd.DataFrame([])
        predicted = [model.predict(vect.transform(content))[0] for content in df]
        result['input'] = df
        result['predicted'] = predicted
        return result
    else:
        print('Invalid Input.')


def to_csv(df, fn):
    """Writes dataframe to csv file."""
    
    df = df.set_index('input')
    with open(fn, 'w+', encoding='latin-1', newline='') as file:
        df.to_csv(file)
    print('Output File: ', os.path.abspath(fn))


if __name__ == '__main__':
    # PARSING ARGUMENTS
    parser = argparse.ArgumentParser(prog='python predict_bulk.py')
    parser.add_argument("model", type=str, action="store", help="Input model.")
    parser.add_argument("vectorizer", type=str, action="store", help="Input vectorizer.")
    parser.add_argument("csv", type=str, action="store", help="Input csv file containing contents to predict.")

    args = parser.parse_args()

    df_out = predict_bulk(args.vectorizer, args.model, pd.read_csv(args.csv, encoding='latin-1'))
    to_csv(df_out, 'predicted.csv')
