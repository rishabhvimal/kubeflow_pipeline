import argparse
import os
import pandas as pd
import gcsfs
from sklearn.model_selection import train_test_split
import datetime
import logging
from sklearn.preprocessing import StandardScaler

def dummy_var(df):
    df_new=pd.get_dummies(df,'Gender')
    return (df_new)

def read_file(path,filename,t_size,target):
    file_path=os.path.join(path,filename)
    t_size=t_size
    df=pd.read_csv(file_path)
    df_dum=dummy_var(df)
    X = df_dum.drop(['User ID',target], axis=1)
    y = df_dum[[target]]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=t_size,random_state=10)
    '''sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)'''
    return (X_train, X_test, y_train, y_test)

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  
  parser = argparse.ArgumentParser(description='Feature Engineering Of the Raw Data')
  
  parser.add_argument('--path', type=str, help='Local or GCS path to the raw file.')
  parser.add_argument('--filename', type=str, help='Filename to be processed.')
  parser.add_argument('--t_size', type=float, default=0.2, help='Test size in percent')
  parser.add_argument('--target', type=str, help='Target variable')

  args = parser.parse_args()

  X_train, X_test, y_train, y_test=read_file(args.path,args.filename,args.t_size,args.target)
  output_path=args.path+'outputs/'
  
  X_tr_file=output_path+'X_train.csv'
  y_tr_file=output_path+'y_train.csv'
  X_te_file=output_path+'X_test.csv'
  y_te_file=output_path+'y_test.csv'
  
  X_train.to_csv(X_tr_file,index=False)
  y_train.to_csv(y_tr_file,index=False)
  X_test.to_csv(X_te_file,index=False)
  y_test.to_csv(y_te_file,index=False)



