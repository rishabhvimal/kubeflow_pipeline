import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import logging
import argparse
import gcsfs
import pickle

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Traiining the Model on data')
    parser.add_argument('--path', type=str, help='Local or GCS path to the training file')
    parser.add_argument('--target', type=str, help='Dependent varaible name.')

    args = parser.parse_args()

    X_train=pd.read_csv(args.path+'outputs/X_train.csv')
    y_train=pd.read_csv(args.path+'outputs/y_train.csv')

    model = LogisticRegression()
    model.fit(X_train,y_train)
    fs = gcsfs.GCSFileSystem()
    pickle.dump(model,fs.open((args.path+'models/model.pkl'),'wb'))






