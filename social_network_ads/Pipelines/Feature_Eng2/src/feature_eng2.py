import argparse
import pandas as pd
import gcsfs
from sklearn.model_selection import train_test_split
import datetime
import logging
from sklearn.preprocessing import StandardScaler
import numpy as np

def standard_scaler(X_train,X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test) 
    return (X_train,X_test)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Feature engineering part 2')
    parser.add_argument('--path', type=str, help='Local or GCS path to the training file')
    
    args = parser.parse_args()
    X_train = pd.read_csv(args.path+'outputs/X_train.csv')
    X_test = pd.read_csv(args.path+'outputs/X_test.csv')
    X_train_scaled,X_test_scaled = standard_scaler(X_train,X_test)

    output_path=args.path+'outputs/'
    X_tr_scaled=output_path+'X_train_scaled.txt'
    X_te_scaled=output_path+'X_test_scaled.txt'

    np.savetxt('X_train_scaled.txt',X_train_scaled)
    np.savetxt('X_test_scaled.txt',X_test_scaled)



