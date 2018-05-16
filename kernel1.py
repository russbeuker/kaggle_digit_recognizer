import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, MaxPooling2D, Flatten
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
import random as rn
from keras.datasets import mnist
import tensorflow as tf
import datetime
import os
import pandas as pd
from kerasbestfit import kbf
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train_file = ".//input//train.csv"
test_file = ".//input//test.csv"
output_file = "submission.csv"
output_model_file = "model.json"
output_model_weights_file = "model_weights.hdf5"
# logging
log_to_file = True
log_file = 'log.txt'
log_mode = 'both'  # screen_only, file_only, both, off

def logmsg(msg=''):
    fmt = "%H:%M:%S"
    s = f'{datetime.datetime.today().strftime(fmt)}: {msg}'
    if log_mode == 'file_only' or log_mode == 'both':
        with open(log_file, "a") as myfile:
            myfile.write(f'{s}\n')
    if log_mode == 'screen_only' or log_mode == 'both':
        print(s)

def load_internal_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    random_seed = np.random.seed(2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed,
        stratify=y_train)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')
    y_test = y_test.astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    y_train = np_utils.to_categorical(y_train, 10)
    y_val = np_utils.to_categorical(y_val, 10)
    # y_test = np_utils.to_categorical(y_test, 10)

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_data(edition: 0):
    logmsg('Loading data.')
    if edition == 0:
        xfile = './/input//x_train.npy'
        yfile = './/input//y_train.npy'
    else:
        fl = '_' + str(edition) + 'x.npy'
        xfile = 'x_train' + fl
        yfile = 'y_train' + fl

    x_train = np.load(file=xfile)
    y_train = np.load(file=yfile)
    # split into train and validation with stratification
    random_seed = np.random.seed(2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed,
                                                      stratify=y_train)
    # set datatypes to float32 and normalize
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = y_train.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)
    x_val = x_val.astype('float32')
    x_val /= 255
    y_val = y_val.astype('float32')
    y_val = np_utils.to_categorical(y_val, 10)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

    # load test data
    x_test = np.load(file='.//input//x_test.npy')
    x_test = x_test.astype('float32')
    x_test /= 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_test = None  # we don't have this data because we don't have the test labels available

    return x_train, y_train, x_val, y_val, x_test, y_test

def predict(x=None, y=None, session_name=None):
    # load the saved model and weights
    with open(session_name + '.json', 'r') as f:
        modelx = model_from_json(f.read())
    modelx.load_weights(session_name + '.hdf5')

    # predict the labels for the x_test images
    test_labels = modelx.predict(x, batch_size=100)
    predicted_classes = np.argmax(test_labels, axis=1)
    if y is not None:
        correct_indices = np.nonzero(predicted_classes == y)[0]
        incorrect_indices = np.nonzero(predicted_classes != y)[0]
        accuracy = len(correct_indices) / (len(correct_indices) + len(incorrect_indices))
        logmsg(f'   Actual prediction for internal data x_test is {accuracy}')
    else:
        # save the results to submission.csv
        submission = pd.DataFrame({
            'ImageID': range(1, 28001),
            'Label': predicted_classes
        })
        logmsg('    Saving predications as submission.csv')
        submission.to_csv('submission.csv', index=False)
        logmsg('    Saved.')

    del modelx

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def train(session_name='test1', metric='val_acc', use_history=False, iterations=1, epochs=2, patience=20,
          snifftest_max_epoch=0,
          snifftest_min_val_acc=0.0, x_train=None, y_train=None, x_val=None, y_val=None, x_test=None, y_test=None,
          shuffle=False, validation_split=0.0, create_confusion_matrix=False, save_best=False, save_path='',
          show_progress=True, format_val_acc='{:1.10f}', max_duration_mins=0, logmsg_callback=None,
          lock_random_seeds=True,
          random_seed=1):

    # define file paths for saving
    results_filename = session_name
    results_path = results_filename
    csvpath = results_path + '.csv'
    my_file = Path(csvpath)

    # get the most recent best val_acc from last run if it exists
    if use_history:
        if my_file.is_file():
            df = pd.read_csv(csvpath)
            best_val_acc_so_far = df['VAL_ACC_BSF'].max()
        else:
            best_val_acc_so_far = 0.0
    else:
        best_val_acc_so_far = 0.0
    s = format_val_acc.format(best_val_acc_so_far)

    # calc timed session
    fmt = "%a %b %d %H:%M:%S"
    if max_duration_mins == 0:
        started_at = 0
        finish_by = 0
    else:
        started_at = datetime.datetime.today()
        finish_by = started_at + datetime.timedelta(minutes=max_duration_mins)
        logmsg(f'Started at {started_at.strftime(fmt)}, finish by {finish_by.strftime(fmt)}')




    logmsg(f'Training {iterations} iterations with {x_train.shape[0]} images. Starting with best val_acc of {s}.')
    iter_best_fit = 0
    for xtr in range(0, iterations):
        K.clear_session()

        if lock_random_seeds:
            np.random.seed(random_seed)
            rn.seed(random_seed)
            tf.set_random_seed(random_seed)

        y_test = None
        #  create model
        #
        # model = Sequential()
        # model.add(Dense(1500, activation='relu', input_shape=(784,), kernel_initializer='he_normal'))
        # model.add(Dropout(0.5))
        # # model.add(BatchNormalization())
        # model.add(Dense(1500, activation='relu', kernel_initializer='he_normal'))
        # model.add(Dropout(0.5))
        # # model.add(BatchNormalization())
        # model.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))
        # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        input_shape = (28, 28, 1)

        # model = Sequential()
        # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape,
        #                  kernel_initializer='he_normal'))
        # model.add(Dropout(0.25))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
        # model.add(Dropout(0.25))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
        # model.add(Dropout(0.25))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Flatten())
        # model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())
        # model.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.50))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.50))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        batch_size = 1000

        #  train the model
        sbest_val_acc_so_far = format_val_acc.format(best_val_acc_so_far)
        logmsg(f'Iteration {xtr} of {iterations}.  Best val_acc so far is {sbest_val_acc_so_far}')

        results, log = kbf.find_best_fit(model=model, metric=metric, xtrain=x_train, ytrain=y_train, xval=x_val,
            yval=y_val, shuffle=shuffle, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
            patience=patience, snifftest_max_epoch=snifftest_max_epoch, snifftest_min_val_acc=snifftest_min_val_acc,
            show_progress=show_progress, progress_val_acc_format=format_val_acc, save_best=save_best,
            save_path=results_path, best_val_acc_so_far=best_val_acc_so_far, finish_by=finish_by,
            logmsg_callback=logmsg_callback
            )
        del model
        # notify if we found a new best val_acc
        if results['best_val_acc'] > best_val_acc_so_far:
            iter_best_fit = xtr
            best_val_acc_so_far = results['best_val_acc']
            sbest_val_acc_so_far = format_val_acc.format(best_val_acc_so_far)
            sbest_epoch = results['best_epoch']
            logmsg(
                f'Iteration {iter_best_fit} has the best val_acc so far with {sbest_val_acc_so_far} found at epoch {sbest_epoch}.')

        print('')

        # generate plot or accuracies and errors
        font1 = {'family': 'serif', 'color': 'darkgreen', 'weight': 'normal', 'size': 10, }
        font2 = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 10, }
        xindent = 1
        yindent = 0.55
        plt.xlim(0.0, epochs - 1.0)
        plt.ylim(0.0, 1.0)
        plt.plot(results['history']['acc'])
        plt.plot(results['history']['val_acc'])
        plt.plot(results['history']['loss'])
        plt.plot(results['history']['val_loss'])
        texttop = 0.1
        plt.title('')
        # plt.yscale('log')
        # plt.semilogy()

        plt.text(xindent, yindent, f'bsf_val_acc={sbest_val_acc_so_far}\n' +
                 f'this_val_acc={results["best_val_acc"]}\n' +
                 f'this_epoch={results["best_epoch"]}\n',
                 fontdict=font1)

        plt.axhline(best_val_acc_so_far, 0, epochs, color='k', linestyle='--')
        plt.axvline(results['best_epoch'], 0, 1, color='k', linestyle='--')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['acc', 'val_acc', 'loss', 'val_loss'], loc='center right')
        plt.show()

        if results['expired']:
            break

    logmsg(f'The best fit was iteration {iter_best_fit} with best val_acc of {sbest_val_acc_so_far} at epoch {sbest_epoch}.')
    logmsg('')




def main():
    my_file = Path(log_file)
    if my_file.is_file():
        os.remove(log_file)
    session_name = 'test1'
    save_model = True

    logmsg('-- LOADING DATA ------------------------')
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(0)
    logmsg(
        f'We will train the model with {x_train.shape[0]} images, and later test the model with {x_test.shape[0]} images.')
    logmsg('-- TRAINING  ------------------------')
    train(session_name=session_name, use_history=False, metric='val_acc', iterations=30, epochs=200, patience=20,
          snifftest_max_epoch=0, snifftest_min_val_acc=0.0,
          x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, shuffle=False,
          validation_split=0, save_best=save_model, save_path='',
          create_confusion_matrix=False, show_progress=True, format_val_acc='{:1.10f}', max_duration_mins=30,
          logmsg_callback=logmsg)
    logmsg('-- PREDICTING ------------------------')
    if save_model:
        predict(x_test, y_test, session_name)

        logmsg('---- PREDICTING INTERNAL ---------------------')
        x_train, y_train, x_val, y_val, x_test, y_test = load_internal_data()
        predict(x_test, y_test, session_name)

logmsg('-- END ------------------------')


if __name__ == "__main__":
    main()

# note: kaggle scored .99114 on a simple unaugmented 10% training/val split that scored 99.047 on my desktop.
# so tomorrow, try some more unaugmented.
#
# kaggle    val_acc     internal
# 0.99185   0.99400     0.9962      - stratified, batch 2000
# 0.99400   0.99452     0.9975      - dropout = 0.5, batch 2000.  wtf kaggle and val_acc matches?
# 0.99342   0.99500     0.9968      - batch 1000


#
#
#
