import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from keras.models import Sequential, Input, Model, model_from_json
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import np_utils
from keras import backend as K
import datetime
import os
import shutil
import pandas as pd
from kerasbestfit import kbf
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train_file = ".//input//train.csv"
test_file = ".//input//test.csv"
output_file = "submission.csv"
output_model_file = "model.json"
output_model_weights_file = "model_weights.hdf5"

mnist_train_dataset = pd.read_csv(train_file, delimiter=',').values
mnist_test_dataset = pd.read_csv(test_file, delimiter=',').values

print(f'train.csv dimensions are {mnist_train_dataset.shape}')
# the first element in each row is the label, followed by 784 pixel values.
# print(f'Here is the first row.\n {mnist_train_dataset[0,:]}')
# first column holds the labels, so let's put those labels in y_all
y_train = mnist_train_dataset[:,0]
print(f'y_train dimensions are {y_train.shape}.  These are the TRAINING labels.')
# the rest of the columns are the pixel data, so let's put that in x_all
x_train = mnist_train_dataset[0:,1:]
print(f'x_train dimensions are {x_train.shape}.  These are the TRAINING images.')
# the rest of the columns are the pixel data, so let's put that in x_all
# we'll use the keras .fit function to split the training data into separate train and validation data sets
y_train = np_utils.to_categorical(y_train, 10)
#set the datatype
x_train = x_train.astype('float32')
#normalize the data so that it is from 0.0 to 1.0
x_train /= 255


# let's see the mnist_test_dataset dimensions
print(f'test.csv dimensions are {mnist_test_dataset.shape}')
# each row is has 784 pixel values.  There is no label, because we aren't supposed to know what digit this is.
# print(f'Here is the first row.\n {mnist_test_dataset[0,:]}')
x_test = mnist_test_dataset
y_test = None
#set the datatype
x_test = x_test.astype('float32')
#normalize the data so that it is from 0.0 to 1.0
x_test /= 255
print(f'x_test dimensions are {x_test.shape}.  These are the TEST images.')
print('------------------------')
print(f'So far, so good. We will train the model with {x_train.shape[0]} images, and later test the model with {x_test.shape[0]} images.')


def predict(filename=None):
    # load the saved model and weights
    with open(filename + '.json', 'r') as f:
        modelx = model_from_json(f.read())
    modelx.load_weights(filename + '.hdf5')

    # predict the labels for the x_test images
    test_labels = modelx.predict(x_test, batch_size=1000)
    pred = np.argmax(test_labels, axis=1)
    # save the results to submission.csv
    submission = pd.DataFrame({
        'ImageID': range(1, 28001),
        'Label': pred
    })
    print('         Saving predications as submission.csv')
    print('')
    print(submission.head())
    print('')
    submission.to_csv('submission.csv', index=False)
    print('         Saved.')


def train(name='test1', metric='val_acc', use_history=False, iterations=1, epochs=2, patience=20, snifftest_max_epoch=0,
          snifftest_min_val_acc=0.0,
          x_train=None, y_train=None, x_val=None, y_val=None, x_test=None, y_test=None, shuffle=False,
          validation_split=0.0,
          create_confusion_matrix=False, save_best=False, save_path='', show_progress=True, format_val_acc='{:1.10f}',
          max_duration_mins=0):
    # define file paths for saving
    results_filename = name
    results_path = results_filename
    csvpath = results_path + '.csv'
    my_file = Path(csvpath)
    # get the most recent best val_acc from last run if it
    if use_history:
        if my_file.is_file():
            df = pd.read_csv(csvpath)
            best_val_acc_so_far = df['VAL_ACC_BSF'].max()
        else:
            best_val_acc_so_far = 0.0
    else:
        best_val_acc_so_far = 0.0
    s = format_val_acc.format(best_val_acc_so_far)
    # print(f'Training {iterations} iterations with {x_train.shape[0]} images and evaluating with {y_val.shape[0]} images. Starting with best val_acc of {s}.')

    fmt = "%a %b %d %H:%M:%S %Y"
    if max_duration_mins == 0:
        started_at = 0
        finish_by = 0
    else:
        started_at = datetime.datetime.today()
        finish_by = started_at + datetime.timedelta(minutes=max_duration_mins)
        print(f'Started at {started_at.strftime(fmt)}, finish by {finish_by.strftime(fmt)}')

    iter_best_fit = 0
    print('-- BEGIN ------------------------')
    for xtr in range(0, iterations):
        K.clear_session()
        y_test = None
        #  create model
        model = Sequential()
        model.add(Dense(1500, activation='relu', input_shape=(784,), kernel_initializer='he_normal'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(1500, activation='relu', input_shape=(784,), kernel_initializer='he_normal'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        batch_size = 500

        #  train the model
        sbest_val_acc_so_far = format_val_acc.format(best_val_acc_so_far)
        print(f'Iteration {xtr} of {iterations}.  Best val_acc so far is {sbest_val_acc_so_far}')
        results, log = kbf.find_best_fit(
            model=model,
            metric=metric,
            xtrain=x_train,
            ytrain=y_train,
            xval=x_val,
            yval=y_val,
            shuffle=shuffle,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            snifftest_max_epoch=snifftest_max_epoch,
            snifftest_min_val_acc=snifftest_min_val_acc,
            show_progress=show_progress,
            progress_val_acc_format=format_val_acc,
            save_best=save_best,
            save_path=results_path,
            best_val_acc_so_far=best_val_acc_so_far,
            started_at=started_at,
            finish_by=finish_by
        )

        # notify if we found a new best val_acc
        if results['best_val_acc'] > best_val_acc_so_far:
            iter_best_fit = xtr
            best_val_acc_so_far = results['best_val_acc']
            sbest_val_acc_so_far = format_val_acc.format(best_val_acc_so_far)
            sbest_epoch = results['best_epoch']
            print(
                f'  Iteration {iter_best_fit} has the best val_acc so far with {sbest_val_acc_so_far} found at epoch {sbest_epoch}.')

        print('')
        if results['expired']:
            break

    print(
        f'  The best fit was iteration {iter_best_fit} with the best val_acc of {sbest_val_acc_so_far} found at epoch {sbest_epoch}.')
    print('')
    print('-- TESTING SAVED MODEL ------------------------')

    if save_best:
        predict('test1')




train(name='test1', use_history=False, metric='val_acc', iterations=10, epochs=200, patience=20,snifftest_max_epoch=0, snifftest_min_val_acc=0.0,
      x_train=x_train, y_train=y_train, x_val=[], y_val=[], x_test=x_test, y_test=y_test, shuffle=True, validation_split=0.15, save_best=True, save_path='',
      create_confusion_matrix=False, show_progress=True, format_val_acc='{:1.10f}', max_duration_mins=5.0)
