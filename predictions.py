from models import load_model
import numpy as np
import pandas as pd
import utils

def predict(sess=None, x=None, y=None, session_name=None):
    modelx = load_model(sess)
    # predict the labels for the x_test images
    test_labels = modelx.predict(x)
    predicted_classes = np.argmax(test_labels, axis=1)
    if y is not None:
        correct_indices = np.nonzero(predicted_classes == y)[0]
        incorrect_indices = np.nonzero(predicted_classes != y)[0]
        accuracy = len(correct_indices) / (len(correct_indices) + len(incorrect_indices))
        sess.log(f'Actual prediction for keras mnist data x_test is {accuracy}')
    else:
        # save the results to submission.csv
        submission = pd.DataFrame({
            'ImageID': range(1, 28001),
            'Label': predicted_classes
        })
        submission.to_csv(sess.full_path + 'submission.csv', index=False)
        sess.log('Saved predictions as submission.csv')
    del modelx
