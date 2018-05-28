import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import os
import timeit
from datetime import datetime
from keras.datasets import mnist
from pandas import read_csv
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.layers import Dense, Dropout, MaxPooling2D, Flatten, Activation, Input
from keras.layers.convolutional import Conv2D
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model, model_from_json
from kerasbestfit import kbf
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)


def log_msg(msg=''):
    fmt = "%H:%M:%S"
    s = f'{datetime.today().strftime(fmt)}: {msg}'
    if log_mode == 'file_only' or log_mode == 'both':
        with open(log_file, "a") as myfile:
            myfile.write(f'{s}\n')
    if log_mode == 'screen_only' or log_mode == 'both':
        print(s)


#####################################################################
# STEP 1 - LOAD AND CLEAN THE TRAINING DATA
#####################################################################
log_file = ".//log.txt"
log_mode = 'both'
model_file = ""
x_train_file_npy = ".//input//x_train.npy"
y_train_file_npy = ".//input//y_train.npy"
x_test_file_npy = ".//input//x_test.npy"
x_train_cleaned_file_npy = ".//input//x_train_cleaned.npy"
y_train_cleaned_file_npy = ".//input//y_train_cleaned.npy"

# convert the kaggle input data csv's to faster .npy. Keep it as uint8 so the files are small as possible
convert_csv_to_npy = True  # set this to False once you have done this once and the files are created
timeit
if convert_csv_to_npy:
    train_file_csv = ".//input//train.csv"
    test_file_csv = ".//input//test.csv"
    mnist_train_dataset = read_csv(train_file_csv, delimiter=',').values
    y_train = mnist_train_dataset[:, 0]
    y_train = y_train.astype('uint8')
    x_train = mnist_train_dataset[0:, 1:]
    x_train = x_train.astype('uint8')
    mnist_test_dataset = read_csv(test_file_csv, delimiter=',').values
    x_test = mnist_test_dataset
    x_test = x_test.astype('uint8')
    np.save(x_train_file_npy, x_train)
    np.save(y_train_file_npy, y_train)
    np.save(x_test_file_npy, x_test)

# now let's load the training data from those fast .npy files
x_train = np.load(file=x_train_file_npy)
y_train = np.load(file=y_train_file_npy)
# we will load the x_test file later, just before we need to do a prediction using our model

# remove inoorrect images from our training data set
# let's take a look at some of these bad images
plt.imshow(x_train[25708].reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
plt.show()
lst = [12817,  # 0's
       60, 191, 2284, 2316, 5275, 7389, 19633, 19979, 24891, 29296, 32565, 38191, 38544, 40339, 41739,  # 1's
       4677, 7527, 9162, 13471, 16598, 20891, 27364,  # 2's
       240, 11593, 11896, 17966, 25708, 28560, 33198, 34477, 36018, 41492,  # 3's
       1383, 6781, 22478, 23604, 26171, 26182, 26411, 18593, 34862, 36051, 36241, 36830, 37544,  # 4's
       456, 2867, 2872, 5695, 6697, 9195, 18319, 19364, 27034, 29253, 35620,  # 5's
       7610, 12388, 12560, 14659, 15219, 18283, 24122, 31649, 40214, 40358, 40653,  # 6's
       6295, 7396, 15284, 19880, 20089, 21423, 25233, 26366, 26932, 27422, 31741,  # 7's
       8566, 10920, 23489, 25069, 28003, 28851, 30352, 30362, 35396, 36984, 39990, 40675, 40868, 41229,  # 8's
       631, 4226, 9943, 14914, 15065, 17300, 18316, 19399, 20003, 20018, 23135, 23732, 29524, 33641, 40881, 41354  # 9's
       ]
x_cleaned = np.delete(x_train, lst, 0)
y_cleaned = np.delete(y_train, lst, 0)
np.save(x_train_cleaned_file_npy, x_cleaned)
np.save(y_train_cleaned_file_npy, y_cleaned)

# up to this point, we we converted our .csv data to .npy and removed bad images

#####################################################################
# STEP 2 - TRAINING
# 1) load data from npy files
# 2) count the training images and check the class balance
# 3) split the training data into TRAINING and VALIDATION datasets
# 4) convert our datasets into the correct data type and normalization
# 5) for x iterations, train the model and save the best result
#####################################################################
do_training = True
if do_training:
    # reload the clean data from those fast .npy files
    x_train = np.load(file=x_train_cleaned_file_npy)
    y_train = np.load(file=y_train_cleaned_file_npy)

    # count the number of training and test images
    log_msg(f'There are {x_train.shape[0]} training images.')

    # class balance
    log_msg('Class balance:')
    unique, counts = np.unique(y_train, return_counts=True)
    log_msg(np.asarray((unique, counts)))

    # plot a barchart showing class balance and mean count
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Class Balance')
    ax.set_ylabel('Count')
    ax.set_xlabel('Class')
    ax.annotate("Mean", xy=(0.012, 0.86), xycoords="axes fraction")
    plt.xticks(unique, unique)
    ax.axhline(counts.mean(), color='black', linewidth=1)
    ax.bar(unique, counts, color='orange', align="center")
    plt.show()

    # let's take a look at the training data shape and format
    log_msg(f'x_train shape is {x_train.shape} of data type {x_train.dtype}.  These are our training IMAGES.')
    log_msg(f'y_train shape is {y_train.shape} of data type {y_train.dtype}.  These are our training LABELS.')

    # let's examine one of the training images. We'll pick the eleventh image in the array
    sample = 10  # change this value to see different images
    plt.title('Sample: %d  Label: %d' % (sample, y_train[sample]))
    plt.imshow(x_train[sample].reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
    plt.show()

    # now let's split up our training data into TRAINING and VALIDATION datasets.
    random_seed = np.random.seed(2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed,
                                                      stratify=y_train)
    log_msg('')
    log_msg('Changing data type to float and normalizing values...')
    # we'll need change our data type to float and normalize the image data
    # set datatypes to float32 and normalize
    x_train = x_train.astype('float32')
    x_train /= 255
    x_val = x_val.astype('float32')
    x_val /= 255
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
    log_msg(f'x_train shape is {x_train.shape} of data type {x_train.dtype}.  These are our TRAINING images.')
    log_msg(f'x_val shape is {x_val.shape} of data type {x_val.dtype}.  These are our VALIDATION images.')

    # and we'll also need to convert our y_train labels to one-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_train = y_train.astype('float32')
    y_val = np_utils.to_categorical(y_val, 10)
    y_val = y_val.astype('float32')
    log_msg(f'y_train shape is {y_train.shape} of data type {y_train.dtype}.  These are our TRAINING labels.')
    log_msg(f'y_val shape is {y_val.shape} of data type {y_val.dtype}.  These are our VALIDATION labels.')

    # run kerasbestfit
    format_metric_val = '{:1.10f}'
    metric = 'val_acc'
    iter_best_fit = 0
    best_metric_val_so_far = 0.0
    for xtr in range(0, 2):
        K.clear_session()

        # now we'll define our model.  We'll use the Keras functional API
        input = Input(shape=(28, 28, 1))
        x1 = Conv2D(32, (5, 5), padding='same', kernel_initializer='he_normal')(input)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)
        x1 = Dropout(0.5)(x1)
        x1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(32, (5, 5), padding='same', kernel_initializer='he_normal')(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(0.5)(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)
        x1 = Dropout(0.5)(x1)
        x1 = Flatten()(x1)
        x1 = Dense(128, activation='relu', kernel_initializer='he_normal')(x1)
        x1 = Dropout(0.5)(x1)
        output = Dense(10, activation='softmax', kernel_initializer='he_normal')(x1)
        model = Model(inputs=[input], outputs=[output])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        results, log = kbf.find_best_fit(model=model, metric=metric, xtrain=x_train, ytrain=y_train, xval=x_val,
                                         yval=y_val, validation_split=0, batch_size=500, epochs=2, patience=5,
                                         snifftest_max_epoch=0,
                                         snifftest_metric_val=0,
                                         show_progress=True,
                                         format_metric_val=format_metric_val,
                                         save_best=True,
                                         save_path=model_file,
                                         best_metric_val_so_far=best_metric_val_so_far,
                                         logmsg_callback=log_msg,
                                         finish_by=0)
        del model
        # notify if we found a new best metric val
        is_best = False
        if metric == 'val_acc':
            is_best = results['best_metric_val'] > best_metric_val_so_far
        elif metric == 'val_loss':
            is_best = results['best_metric_val'] < best_metric_val_so_far
        if is_best:
            iter_best_fit = xtr
            best_metric_val_so_far = results['best_metric_val']
            sbest_metric_val_so_far = metric + '=' + format_metric_val.format(best_metric_val_so_far)
            sbest_epoch = results['best_epoch']
            s = f'NEW BEST: {sbest_metric_val_so_far}\n'
            log_msg(s)

#############################################################################################
# STEP 3 - PREDICTION
# 1) load saved model
# 2) plot the first image in the test set
# 3) run prediction on test images and save as submission.csv
# 4) run a prediction on the built-in MNIST test dataset to see how well the model performs
#############################################################################################
do_predictions = False
if do_predictions:
    # load test data
    x_test = np.load(file=x_test_file_npy)
    x_test = x_test.astype('float32')
    x_test /= 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_test = None  # we don't have this data because we don't have the test labels available

    # load saved model
    with open('.//model.json', 'r') as f:
        modelx = model_from_json(f.read())
    modelx.load_weights('.//model.hdf5')

    # predict a single test image
    log_msg('Predicting a single test image...')
    img = x_test[0]
    plt.imshow(img.reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
    plt.show()
    img = img.reshape(1, 28, 28, 1)
    prediction = modelx.predict([img])
    log_msg(f'Predicted class probabilities: {prediction[0]}')
    predicted_classes = np.argmax(prediction[0], axis=0)
    log_msg(f'Predicted digit: {predicted_classes}')

    # now let's predict all the test images and save the results to submission.csv
    log_msg('Predicting all test images...')
    test_labels = modelx.predict(x_test)
    predicted_classes = np.argmax(test_labels, axis=1)
    submission = pd.DataFrame({
        'ImageID': range(1, 28001),
        'Label': predicted_classes
    })
    submission.to_csv('submission.csv', index=False)
    log_msg('Saved predictions as submission.csv')

    # try the model with the built-in Keras MNIST dataset
    log_msg('')
    log_msg('Trying the model with the Keras built-in MNIST test dataset...')
    (x_keras_train, y_keras_train), (x_keras_test, y_keras_test) = mnist.load_data()
    x_keras_test = x_keras_test.astype('float32')
    y_keras_test = y_keras_test.astype('float32')
    x_keras_test /= 255
    x_keras_test = x_keras_test.reshape(x_keras_test.shape[0], 28, 28, 1)
    # y_keras_test = np_utils.to_categorical(y_keras_test, 10)
    test_labels = modelx.predict(x_keras_test)
    predicted_classes = np.argmax(test_labels, axis=1)
    correct_indices = np.nonzero(predicted_classes == y_keras_test)[0]
    incorrect_indices = np.nonzero(predicted_classes != y_keras_test)[0]
    accuracy = len(correct_indices) / (len(correct_indices) + len(incorrect_indices))
    log_msg(f'Prediction is {accuracy}')

    del modelx


