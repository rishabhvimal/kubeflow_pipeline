import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.metrics import r2_score,accuracy_score
from sklearn.metrics import confusion_matrix
import argparse
import gcsfs
import os
from tensorflow.python.lib.io import file_io
import json

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Testing the Model on test data.')
    parser.add_argument('--path', type=str, help='Local or GCS path to the test file')

    args = parser.parse_args()

    X_test = pd.read_csv(args.path + 'outputs/X_test.csv')
    y_test=pd.read_csv(args.path+'outputs/y_test.csv')

    fs = gcsfs.GCSFileSystem(token='cloud')
    model = pickle.load(fs.open((args.path + 'models/model.pkl'), 'rb'))

    Predict=model.predict(X_test)
    y_test['Predicted'] = Predict
    y_test.to_csv(args.path+'test_pred/ActualPredict.csv')

    vocab = list(y_test['Purchased'].unique())
    cm = confusion_matrix(y_test['Purchased'], y_test['Predicted'], labels=vocab)
    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))

    df_cm = pd.DataFrame(data, columns=['Purchased', 'Predicted', 'count'])
    cm_file = os.path.join(args.path, 'confusion_matrix.csv')
    with file_io.FileIO(cm_file, 'w') as f:
        df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)

    metadata = {
        'outputs' : [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': cm_file,
            # Convert vocab to string because for bealean values we want "True|False" to match csv data.
            'labels': list(map(str, vocab)),
        }]
    }
    with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    print("Test accuracy is:", np.round(np.sqrt(accuracy_score(y_test['Purchased'],y_test['Predicted'])), 3))
    print("Test r2_score is:", np.round(np.sqrt(r2_score(y_test['Purchased'], y_test['Predicted'])), 3))
