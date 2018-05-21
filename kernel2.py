# custom
import globals
from data_loader import load_keras_mnist_data, load_kaggle_mnist_data
from utils import logmsg, delete_log_file
from models import create_model_1
from predictions import predict
from sessions import TrainingSession

# native
import numpy as np
import tensorflow as tf
from keras import backend as K
import random as rn
import datetime
from kerasbestfit import kbf

# ----------------------------------------------------------------------------------------------------------------------
def train(session=None, metric='val_acc', iterations=1, epochs=2, patience=20,
          snifftest_max_epoch=0, snifftest_metric_val=0.0, x_train=None, y_train=None, x_val=None, y_val=None,
          shuffle=False, validation_split=0.0, save_best=False, save_path='',
          show_progress=True, format_metric_val='{:1.10f}', max_duration_mins=0, logmsg_callback=None,
          lock_random_seeds=True, random_seed=1, progress_callback=None):

    # init the bestsofar metric
    if metric=='val_acc':
        best_metric_val_so_far = 0.0
    elif metric=='val_loss':
        best_metric_val_so_far = 100.0
    sbest_metric_val_so_far = metric + '=' + format_metric_val.format(best_metric_val_so_far)

    session.log(f'Training {iterations} iterations with {x_train.shape[0]} images. Starting with best {sbest_metric_val_so_far}.')

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
            yval=y_val, shuffle=shuffle, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
            patience=patience, snifftest_max_epoch=snifftest_max_epoch, snifftest_metric_val=snifftest_metric_val,
            show_progress=show_progress, format_metric_val=format_metric_val, save_best=save_best,
            save_path=save_path, best_metric_val_so_far=best_metric_val_so_far, finish_by=finish_by,
            logmsg_callback=logmsg_callback, progress_callback=progress_callback
            )

        del model

        # notify if we found a new best metric val
        is_best=False
        if metric=='val_acc':
            is_best = results['best_metric_val'] > best_metric_val_so_far
        elif metric=='val_loss':
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




def on_progress(epoch, acc, loss, val_acc, val_loss):
    # print(epoch, acc, loss, val_acc, val_loss)
    # save this to disk so other app can display graph
    return

def main():
    # create session
    session_name = 'test1'
    session_parent_dir = f'.//sessions//'
    sess = TrainingSession(session_name, session_parent_dir, log_mode='both', timestamped_folder=True)

    sess.log('-- LOADING DATA ------------------------')
    x_train, y_train, x_val, y_val, x_test, y_test = load_kaggle_mnist_data(sess, 0, True)
    sess.log(f'Training model with {x_train.shape[0]} images.  Testing the model with {x_test.shape[0]} images.')

    sess.log('-- TRAINING  ------------------------')
    train(session=sess, metric='val_loss',  x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, shuffle=False,
          validation_split=0, save_best=True, save_path=sess.full_path, show_progress=True,
          format_metric_val='{:1.10f}', logmsg_callback=sess.log, progress_callback=on_progress,
          max_duration_mins = 480,
          iterations = 1,
          epochs = 2,
          patience = 20,
          snifftest_max_epoch = 0,
          snifftest_metric_val = 100.0
    )

    sess.log('-- PREDICTING KAGGLE ------------------------')
    predict(sess, x=x_test, y=y_test)

    sess.log('-- PREDICTING KERAS INTERNAL ------------------')
    x_train, y_train, x_val, y_val, x_test, y_test = load_keras_mnist_data(sess)
    predict(sess, x=x_test, y=y_test)

    sess.log('-- END ------------------------')

if __name__ == "__main__":
    main()
