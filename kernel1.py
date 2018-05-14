import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from keras.models import Sequential, Input, Model, model_from_json
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from imgaug import augmenters as iaa
import datetime
import os
import shutil
import pandas as pd
from kerasbestfit import kbf
from pathlib import Path
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train_file = ".//input//train.csv"
test_file = ".//input//test.csv"
output_file = "submission.csv"
output_model_file = "model.json"
output_model_weights_file = "model_weights.hdf5"
#logging
log_to_file=True
log_file='log.txt'
log_mode='both'  #screen_only, file_only, both, off

def logmsg(msg=''):
    fmt = "%H:%M:%S"
    s = f'{datetime.datetime.today().strftime(fmt)}: {msg}'
    if log_mode=='file_only' or log_mode=='both':
        with open(log_file, "a") as myfile:
            myfile.write(f'{s}\n')
    if log_mode=='screen_only' or log_mode=='both':
        print(s)

def augment_images(x_train=None, y_train=None, multiplier=2):
    logmsg('Augmenting images...')
    file_x_train = f'.//x_train_{str(multiplier)}x.npy'
    file_y_train = f'.//y_train_{str(multiplier)}x.npy'
    x_train_inflated = x_train
    y_train_inflated = y_train
    for x in range(0, multiplier - 2):
        x_train_inflated = np.concatenate((x_train_inflated, x_train), axis=0)
        y_train_inflated = np.concatenate((y_train_inflated, y_train), axis=0)
    logmsg(f'Started with: {x_train.shape}, adding an additional augmented: {x_train_inflated.shape}')

    # randomly augment the images
    seq = iaa.OneOf([
                       iaa.Sequential(
                            [
                                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                                iaa.Affine(rotate=(0.5, 1.5)),
                                iaa.CropAndPad(percent=(-0.25, 0.25))
                            ]
                        ),
                        iaa.Sequential(
                            [
                                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                                iaa.Affine(rotate=(0.5, 1.5)),
                                iaa.CropAndPad(percent=(-0.25, 0.25)),
                                iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
                            ]
                        ),
                        iaa.Sequential(
                            [
                                iaa.CoarseDropout(0.10, size_percent=0.33),
                                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                                iaa.Affine(rotate=(0.5, 1.5)),
                                iaa.CropAndPad(percent=(-0.25, 0.25))
                            ]
                        )
                ])
    train_inflated = x_train_inflated.reshape(x_train_inflated.shape[0], 28, 28, 1)
    x_train_aug = seq.augment_images(train_inflated)
    # insert the original 60k images and labels at the beginning
    x_train_augx = x_train_aug.reshape(x_train_aug.shape[0], 784)
    x_train = np.concatenate((x_train, x_train_augx), axis=0)
    y_train = np.concatenate((y_train, y_train_inflated), axis=0)
    logmsg('Augmention complete.')

    # save to file
    np.save(file_x_train, x_train)
    np.save(file_y_train, y_train)
    logmsg('Saved.')

    return x_train, y_train

def load_data_kaggle(augment=False, augment_multiplier=2, load_from_file=False):
    logmsg('Loading Kaggle data.')
    #load test data
    mnist_test_dataset = pd.read_csv(test_file, delimiter=',').values
    x_test = mnist_test_dataset
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = None  # we don't have this data because we don't have the test labels available

    already_loaded = True
    if load_from_file:
        file_x_train = f'.//x_train_{str(augment_multiplier)}x.npy'
        file_y_train = f'.//y_train_{str(augment_multiplier)}x.npy'
        my_file_x_train = Path(file_x_train)
        my_file_y_train = Path(file_y_train)
        if (my_file_x_train.is_file() and my_file_y_train.is_file()):
            logmsg(f'Loading from {my_file_x_train} and {file_y_train}...')
            x_train = np.load(file=file_x_train)
            y_train = np.load(file=file_y_train)
            y_train = np_utils.to_categorical(y_train, 10)
            x_train = x_train.astype('float32')
            x_train /= 255
            y_test = None
            return x_train, y_train, x_test, y_test
        else:
            already_loaded = False

    if not load_from_file or not already_loaded:
        mnist_train_dataset = pd.read_csv(train_file, delimiter=',').values
        y_train = mnist_train_dataset[:,0]
        x_train = mnist_train_dataset[0:,1:]
        x_train = x_train.astype('float32')
        x_train /= 255
        if augment:
            x_train, y_train = augment_images(x_train, y_train, augment_multiplier)
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = None
        return x_train, y_train, x_test, y_test


def predict(x=None, y=None, session_name=None):
    # load the saved model and weights
    with open(session_name + '.json', 'r') as f:
        modelx = model_from_json(f.read())
    modelx.load_weights(session_name + '.hdf5')

    # predict the labels for the x_test images
    test_labels = modelx.predict(x, batch_size=1000)
    pred = np.argmax(test_labels, axis=1)
    # save the results to submission.csv
    submission = pd.DataFrame({
        'ImageID': range(1, 28001),
        'Label': pred
    })
    logmsg('         Saving predications as submission.csv')
    logmsg('')
    logmsg(submission.head())
    logmsg('')
    submission.to_csv('submission.csv', index=False)
    logmsg('         Saved.')


def train(session_name='test1', metric='val_acc', use_history=False, iterations=1, epochs=2, patience=20, snifftest_max_epoch=0,
          snifftest_min_val_acc=0.0, x_train=None, y_train=None, x_val=None, y_val=None, x_test=None, y_test=None,
          shuffle=False, validation_split=0.0, create_confusion_matrix=False, save_best=False, save_path='',
          show_progress=True, format_val_acc='{:1.10f}', max_duration_mins=0, logmsg_callback=None):
    # define file paths for saving
    results_filename = session_name
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

    logmsg(f'Training {iterations} iterations with {x_train.shape[0]} images. Starting with best val_acc of {s}.')

    fmt = "%a %b %d %H:%M:%S"
    if max_duration_mins == 0:
        started_at = 0
        finish_by = 0
    else:
        started_at = datetime.datetime.today()
        finish_by = started_at + datetime.timedelta(minutes=max_duration_mins)
        logmsg(f'Started at {started_at.strftime(fmt)}, finish by {finish_by.strftime(fmt)}')

    iter_best_fit = 0
    for xtr in range(0, iterations):
        K.clear_session()
        y_test = None
        #  create model
        model = Sequential()
        model.add(Dense(1500, activation='relu', input_shape=(784,), kernel_initializer='he_normal'))
        model.add(Dropout(0.5))
        # model.add(BatchNormalization())
        model.add(Dense(1500, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.5))
        # model.add(BatchNormalization())
        model.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        batch_size = 1000

        #  train the model
        sbest_val_acc_so_far = format_val_acc.format(best_val_acc_so_far)
        logmsg(f'Iteration {xtr} of {iterations}.  Best val_acc so far is {sbest_val_acc_so_far}')
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
            finish_by=finish_by,
            logmsg_callback=logmsg_callback
        )

        # notify if we found a new best val_acc
        if results['best_val_acc'] > best_val_acc_so_far:
            iter_best_fit = xtr
            best_val_acc_so_far = results['best_val_acc']
            sbest_val_acc_so_far = format_val_acc.format(best_val_acc_so_far)
            sbest_epoch = results['best_epoch']
            logmsg(
                f'  Iteration {iter_best_fit} has the best val_acc so far with {sbest_val_acc_so_far} found at epoch {sbest_epoch}.')

        print('')

        if results['expired']:
            break

    logmsg(f'The best fit was iteration {iter_best_fit} with the best val_acc of {sbest_val_acc_so_far} found at epoch {sbest_epoch}.')
    logmsg('')


def main():
    my_file = Path(log_file)
    if my_file.is_file():
        os.remove(log_file)
    session_name='test1'
    save_model=True

    logmsg('-- LOADING DATA ------------------------')
    x_train, y_train, x_test, y_test = load_data_kaggle(augment=True, augment_multiplier=6, load_from_file=True)
    logmsg(f'We will train the model with {x_train.shape[0]} images, and later test the model with {x_test.shape[0]} images.')

    logmsg('')
    logmsg('-- TRAINING  ------------------------')
    train(session_name=session_name, use_history=False, metric='val_acc', iterations=100, epochs=200, patience=20,snifftest_max_epoch=0, snifftest_min_val_acc=0.0,
        x_train=x_train, y_train=y_train, x_val=[], y_val=[], x_test=x_test, y_test=y_test, shuffle=True, validation_split=0.15, save_best=save_model, save_path='',
        create_confusion_matrix=False, show_progress=True, format_val_acc='{:1.10f}', max_duration_mins=5, logmsg_callback=logmsg)

    logmsg('-- PREDICTING ------------------------')

    if save_model:
        predict(x_test, y_test, session_name)

    logmsg('-- END ------------------------')

if __name__ == "__main__":
    main()