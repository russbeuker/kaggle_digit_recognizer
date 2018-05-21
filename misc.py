# batch_size = rn.randint(800, 2000)
# print(f'batch_size={batch_size}')

# 0.60 = 0.01427, .9971, batch 1500
# 0.55 = 0.01294, .9974, batch 1500
# 0.45 = 0.01156, .9976, batch 1500
# 0.40 = 0.01222, .9973, batch 1500

# 0.50 = 0.01105, .9979, batch 1500
# 0.50 = 0.01030, .9979, batch 1000
# 0.50 = 0.00978, .9984, batch 500 - overfitting? = 0.99542 kaggle - high

# 0.50 = 0.010369, .9979, batch 300 - overfitting?

# 0.50 = 0.01272, .9984, batch 100 - overfitting?
# 0.50 = 0.00995, .9979, batch 3000 - hmmm


# next, find best batchsize for the 0.50 dropout.  automate and graph it.
# plot scatterplot.  batch vs dropout vs val_loss, batch vs droput vs val_acc
# batchsize   dropout     val_loss    val_acc
# 1500        0.50        0.1105       .9979
#
# batchsize=range(16, 3000, step 100)
# droprate=range(0, 0.9, step 0.05)
# take average of 5 tries per combination of batchsize and droprate
# snifftests off
# gridsearch would be 600 combinations * 5 tries each
# lock random seeds
# needs to be restartable to retry the most recent incomplete combination would be nice.
















#ideas
# try reducing color depth to 16 grayscale
# try increasing color depth to 16384
# try jittering x,y postion of entire image when training and when predicting (eye sacades)
# hyperprops - split random seed, batch_size, dropout

# note: kaggle scored .99114 on a simple unaugmented 10% training/val split that scored 99.047 on my desktop.
# so tomorrow, try some more unaugmented.
#
# kaggle    val_acc     val_loss    val_acc_int
# 0.99185   0.99400                 0.9962      - stratified, batch 2000
# 0.99400   0.99452                 0.9975      - dropout = 0.5, batch 2000.  wtf kaggle and val_acc matches?
# 0.99342   0.99500                 0.9968      - batch 1000
# 0.99385               0.02140     0.9977
# 0.99242               0.02098     0.9968      - batch 1132
# 0.99342               0.019411     0.9978
# 0.99471               0.019385     0.9975      = batch 1000, model A1
# 0.99457               0.019566     0.9979      = batcj 1000, model A1
# 0.99528               0.019300     0.9972      = batch 1500
# 0.99400               0.018517     99.79
# 0.99442               0.017950     99.79       = batch 1500
# 0.94280               0.017762     99.80       = batch 1500
# 0.99385               0.01611      99.79
# 0.99471    0.99643                 99.79

# 0.99385               0.01728      99.77       - model A, batch 1500, data not cleaned, split randseed=2
# 0.99457               0.01041      99.82       - model A, batch 1500, cleaned data, split randseed=2
# 0.99342               0.00816      99.81       - model A, batch 1500, cleaned data, split randseed=8
# 0.99285               0.01405      99.67       - model A, batch 1500, cleaned data, split randseed=2, droprate=0.2



#
# Model A!
# dropout = 0.5
# input_shape = (28, 28, 1)
# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', kernel_initializer='he_normal',
#                  input_shape=(28, 28, 1)))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', kernel_initializer='he_normal'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(dropout))

# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', kernel_initializer='he_normal'))
# model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', kernel_initializer='he_normal'))
# model.add(Dropout(dropout))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(dropout))

# model.add(Flatten())
# model.add(Dense(128, activation="relu", kernel_initializer='he_normal'))
# model.add(Dropout(dropout))
# model.add(Dense(10, activation="softmax", kernel_initializer='he_normal'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#
# # generate plot or accuracies and errors
# font1 = {'family': 'serif', 'color': 'darkgreen', 'weight': 'normal', 'size': 10, }
# font2 = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 10, }
# xindent = 1
# yindent = 0.55
# plt.xlim(0.0, results['final_epoch'] + 3.0)
# plt.ylim(0.0, 1.0)
# plt.plot(results['history']['acc'])
# plt.plot(results['history']['val_acc'])
# plt.plot(results['history']['loss'])
# plt.plot(results['history']['val_loss'])
# texttop = 0.1
# plt.title(metric + '=' + format_metric_val.format(results['best_metric_val']))
# # plt.yscale('log')
# # plt.semilogy()
# #
# # plt.text(xindent, yindent, f'bsf_val_acc={sbest_val_acc_so_far}\n' +
# #          f'this_val_acc={results["best_val_acc"]}\n' +
# #          f'this_epoch={results["best_epoch"]}\n',
# #          f'this_val_acc={results["best_val_acc"]}\n' +
# #          f'batch_size={batch_size]}\n',
# #          fontdict=font1)
#
# plt.axhline(best_metric_val_so_far, 0, epochs, color='k', linestyle='--')
# plt.axvline(results['best_epoch'], 0, 1, color='k', linestyle='--')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc', 'loss', 'val_loss'], loc='center right')
# plt.show()
