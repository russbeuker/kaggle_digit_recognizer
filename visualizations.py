import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec

def display_mnist_digit(images, labels, num):
    image = images[num].reshape([28, 28])
    label = labels[num].argmax(axis=0)
    plt.title('Sample: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_mnist_digits(x, y, start, finish):
    for i in range(start, finish):
        q = finish - start
        if int(q/5) == q/5:
            w = int(q/5)
        else:
            w = int(q / 5) + 1
        plt.subplot(w, 5, i - start +1)
        label = y[i].argmax(axis=0)
        plt.title('%i: %d' % (i, label))
        plt.imshow(x[i].reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    plt.show()

def display_mnist_digits_packed_100(x, y, filter=None, title=None):
    if filter != None:
        y = np.where(y == filter)[0]
        x = x[y]
    cells = x.shape[0]
    if cells > 100: cells = 100
    if int(cells / 10) == cells / 10:
        rows = int(cells/ 10)
    else:
        rows = int(cells/ 10) + 1
    plt.figure(figsize=(10, 10))
    gs1 = gridspec.GridSpec(10, 10)
    gs1.update(wspace=0.001, hspace=0.001)  # set the spacing between axes.
    for i in range(0, cells):
        ax1 = plt.subplot(gs1[i])
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        if filter == None:
            label = y[i]
            plt.text(1,5,label, color='red')
        plt.imshow(x[ i].reshape([28, 28]), cmap=plt.get_cmap('gray_r'), vmax=255)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    plt.suptitle(title, fontsize=20)
    plt.show()

def plot_confusion_matrix(model, test_images, test_labels, title=None):
    Y_pred = model.predict(test_images)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(test_labels, axis=1)
    cm = confusion_matrix(Y_true, Y_pred_classes)
    classes = range(10)
    normalize = False  # set to true to give percentages instead of instances
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # val = cm[i, j]
        # val = 100.0 * val/cm[i, :].sum()
        # print(cm[i, j], val)
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
