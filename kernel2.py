import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime
import random as rn
from scipy import arange
import numpy as np
import tensorflow as tf
from keras import backend as K
from kerasbestfit import kbf
from data_loader import load_keras_mnist_data, load_kaggle_mnist_data
from models import create_model_1
from predictions import predict
from sessions import TrainingSession
from pathlib import Path
import pandas as pd
from pandas import DataFrame
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def on_progress(epoch, acc, loss, val_acc, val_loss):
    # print(epoch, acc, loss, val_acc, val_loss)
    # save this to disk so other app can display graph
    return

def walk(iterations=1, epochs=2, patience=2, metric='val_acc', format_metric_val='{:1.10f}',
         snifftest_max_epoch=0, snifftest_metric_val=0.0
         ):
    # create session
    session_name = 'test1'
    session_parent_dir = f'.//sessions//'
    sess = TrainingSession(session_name, session_parent_dir, log_mode='both', timestamped_folder=False, delete_folder=False)

    best_metric_val_so_far = 0.0
    if metric == 'val_loss':
        best_metric_val_so_far = 100.0

    # get the most recent state of the test
    my_file = Path(sess.state_path)
    if my_file.is_file():
        df = pd.read_csv(sess.state_path)
        starting_batchsize = int(df['BATCHSIZE'])
        starting_dropout = float(df['DROPOUT'])
        best_metric_val_so_far = float(df['BEST_METRIC_VAL_SO_FAR'])
    else:
        starting_batchsize = 100
        starting_dropout = 0.0

    # starting_batchsize = 1500
    for batch_size in arange(starting_batchsize, 5500, 500):
        starting_batchsize = 100

        for dropout in arange(starting_dropout, 0.95, 0.05):

            # note: resuming does not work.  You need to start it at a certain starting_batchsize and starting dropout, then reset it to 100, 0
            # does the following 2 lines work?
            starting_dropout = 0.0

            sess.log('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            s = f'Trying: Batchsize={batch_size}, Dropout={dropout}'
            sess.log(s)

            # print(batch_size, dropout)
            # continue

            # save state
            sess.log('Saving state.')
            state_log = []
            state = {}
            state['BATCHSIZE'] = batch_size
            state['DROPOUT'] = dropout
            state['BEST_METRIC_VAL_SO_FAR'] = best_metric_val_so_far
            state_log.append(state)
            df = DataFrame.from_records(state_log, columns=state.keys())
            my_file = Path(sess.state_path)
            if my_file.is_file():
                os.remove(sess.state_path)
            df.to_csv(sess.state_path, header=True)

            avg_metric=0
            iter_best_fit = 0
            for xtr in range(0, iterations + 1):
                K.clear_session()
                x_train, y_train, x_val, y_val, x_test, y_test = load_kaggle_mnist_data(sess, 0, True)
                model = create_model_1(dropout=dropout)
                sbest_metric_val_so_far = metric + '=' + format_metric_val.format(best_metric_val_so_far)
                sess.log(f'Iteration {xtr} of {iterations}.  Best {sbest_metric_val_so_far}')
                results, log = kbf.find_best_fit(model=model, metric=metric, xtrain=x_train, ytrain=y_train, xval=x_val,
                                                 yval=y_val, validation_split=0,batch_size=batch_size, epochs=epochs,
                                                 patience=patience, snifftest_max_epoch=snifftest_max_epoch,
                                                 snifftest_metric_val=snifftest_metric_val,
                                                 show_progress=True, format_metric_val=format_metric_val,
                                                 save_best=True,
                                                 save_path=sess.full_path,
                                                 best_metric_val_so_far=best_metric_val_so_far,
                                                 finish_by=0,
                                                 logmsg_callback=sess.log, progress_callback=on_progress)

                del model
                avg_metric = avg_metric + results['best_metric_val']
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
                    s = f'NEW BEST: Batchsize={batch_size}, Dropout={dropout}, Iteration={iter_best_fit}, Result={sbest_metric_val_so_far}'
                    sess.log(s)
                sess.log('')
            avg_val = (avg_metric / (iterations + 1))
            sval = format_metric_val.format(avg_val)
            sess.log(f'>>>> AVERAGE METRIC: {sval}')
            sess.log('')

            # save result
            sess.log('Saving result.')
            result_log = []
            result = {}
            result['BATCHSIZE'] = batch_size
            result['DROPOUT'] = dropout
            result['AVG_METRIC_VAL'] = avg_val
            result_log.append(result)
            df = DataFrame.from_records(result_log, columns=result.keys())
            my_file = Path(sess.results_path)
            if my_file.is_file():
                df.to_csv(sess.results_path, mode='a', header=False)
            else:
                df.to_csv(sess.results_path)










# ----------------------------------------------------------------------------------------------------------------------
def train(session=None, metric='val_acc', iterations=1, epochs=2, patience=5,
          snifftest_max_epoch=0, snifftest_metric_val=0.0, x_train=None, y_train=None, x_val=None, y_val=None,
          shuffle=False, validation_split=0.0, save_best=False, save_path='',
          show_progress=True, format_metric_val='{:1.10f}', max_duration_mins=0, logmsg_callback=None,
          lock_random_seeds=True, random_seed=1, progress_callback=None):
    # init the bestsofar metric
    if metric == 'val_acc':
        best_metric_val_so_far = 0.0
    elif metric == 'val_loss':
        best_metric_val_so_far = 100.0
    sbest_metric_val_so_far = metric + '=' + format_metric_val.format(best_metric_val_so_far)

    session.log(
        f'Training {iterations} iterations with {x_train.shape[0]} images. Starting with best {sbest_metric_val_so_far}.')

    # init timed session
    fmt = "%a %b %d %H:%M:%S"
    if max_duration_mins == 0:
        started_at = 0
        finish_by = 0
    else:
        started_at = datetime.datetime.today()
        finish_by = started_at + datetime.timedelta(minutes=max_duration_mins)
        session.log(f'Started at {started_at.strftime(fmt)}, finish by {finish_by.strftime(fmt)}')

    # lock random seeds
    if lock_random_seeds:
        np.random.seed(random_seed)
        rn.seed(random_seed)
        tf.set_random_seed(random_seed)

    iter_best_fit = 0
    for xtr in range(0, iterations):
        K.clear_session()

        model = create_model_1()

        batch_size = 3000

        #  train the model
        sbest_metric_val_so_far = metric + '=' + format_metric_val.format(best_metric_val_so_far)
        session.log(f'Iteration {xtr} of {iterations}.  Best {sbest_metric_val_so_far}')

        results, log = kbf.find_best_fit(model=model, metric=metric, xtrain=x_train, ytrain=y_train, xval=x_val,
                                         yval=y_val, shuffle=shuffle, validation_split=validation_split,
                                         batch_size=batch_size, epochs=epochs,
                                         patience=patience, snifftest_max_epoch=snifftest_max_epoch,
                                         snifftest_metric_val=snifftest_metric_val,
                                         show_progress=show_progress, format_metric_val=format_metric_val,
                                         save_best=save_best,
                                         save_path=save_path, best_metric_val_so_far=best_metric_val_so_far,
                                         finish_by=finish_by,
                                         logmsg_callback=logmsg_callback, progress_callback=progress_callback
                                         )

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
            session.log(
                f'Iteration {iter_best_fit} has the best val_acc so far with {sbest_metric_val_so_far} found at epoch {sbest_epoch}.')

        session.log('')

        if results['expired']:
            break

    sbest_metric_val_so_far = metric + '=' + format_metric_val.format(best_metric_val_so_far)
    session.log(f'The best fit was iteration {iter_best_fit} with {sbest_metric_val_so_far} at epoch {sbest_epoch}.')
    session.log('')



# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def main():
    walk(iterations=3, epochs=200, patience=10, metric='val_acc', format_metric_val='{:1.10f}',
         snifftest_max_epoch=0, snifftest_metric_val=0)

    quit()

    # create session
    session_name = 'test1'
    session_parent_dir = f'.//sessions//'
    sess = TrainingSession(session_name, session_parent_dir, log_mode='both', timestamped_folder=True)

    sess.log('-- LOADING DATA ------------------------')
    x_train, y_train, x_val, y_val, x_test, y_test = load_kaggle_mnist_data(sess, 0, True)
    sess.log(f'Training model with {x_train.shape[0]} images.  Testing the model with {x_test.shape[0]} images.')

    sess.log('-- TRAINING  ------------------------')
    train(session=sess, metric='val_loss', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, shuffle=False,
          validation_split=0, save_best=True, save_path=sess.full_path, show_progress=True,
          format_metric_val='{:1.10f}', logmsg_callback=sess.log, progress_callback=on_progress,
          max_duration_mins=480,
          iterations=1,
          epochs=2,
          patience=20,
          snifftest_max_epoch=0,
          snifftest_metric_val=100.0
          )

    sess.log('-- PREDICTING KAGGLE ------------------------')
    predict(sess, x=x_test, y=y_test)

    sess.log('-- PREDICTING KERAS INTERNAL ------------------')
    x_train, y_train, x_val, y_val, x_test, y_test = load_keras_mnist_data(sess)
    predict(sess, x=x_test, y=y_test)

    sess.log('-- END ------------------------')

if __name__ == "__main__":
    main()
