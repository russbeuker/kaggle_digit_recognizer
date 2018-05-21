import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from pandas import read_csv
from sklearn.model_selection import train_test_split


# convert the kaggle data csv's to faster .npy
def convert_kaggle_csv_to_pnp():
    train_file_csv = ".//input//train.csv"
    test_file_csv = ".//input//test.csv"
    x_train_file_npy = ".//input//x_train.npy"
    y_train_file_npy = ".//input//y_train.npy"
    x_test_file_npy = ".//input//x_test.npy"
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


# loads and returns the keras mnist data as npy
def load_keras_mnist_data(sess: None):
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
    return x_train, y_train, x_val, y_val, x_test, y_test


# loads and returns the kaggle data as npy
def load_kaggle_mnist_data(sess: None, edition: 0, use_clean: False):
    sess.log('Loading data.')
    if edition == 0:
        if use_clean:
            sess.log('Using cleaned training data.')
            xfile = ".//input//x_train_cleaned.npy"
            yfile = ".//input//y_train_cleaned.npy"
        else:
            sess.log('Using original training data.')
            xfile = './/input//x_train.npy'
            yfile = './/input//y_train.npy'
    else:
        fl = '_' + str(edition) + 'x.npy'
        xfile = 'x_train' + fl
        yfile = 'y_train' + fl
    x_train = np.load(file=xfile)
    y_train = np.load(file=yfile)

    # remove bad data from the train dataset

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
