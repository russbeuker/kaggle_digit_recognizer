import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import numpy as np
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def logmsg(msg=''):
    fmt = "%H:%M:%S"
    s = f'{datetime.datetime.today().strftime(fmt)}: {msg}'
    print(s)

def plot_mnist_digits_packed(x, y, max_length=100, filter=None, show=False, save=True, show_index=False):
    # this will plot each digit into a .png that is 10 columns wide
    # x - this is the x_train data (features) in shape ex. 60000,784, datatype=uint8, pixel values 0 to 255
    # y - this is the y_train data (labels) in shape ex. 60000,1
    # max_length - this the the total number of images that will be plotted.
    # filter - this is the digit that it will plot to the .png.  So set it to 0 if you want to save zero's to a png
    #   set it to None to plot all digit types instead.
    # show - set this to True to make the plot visible
    # save - set this to True to save the .png's in the current directory
    # show_index - put the numpy array index number on each image

    # if a filter is set, remove all digits that do not match the filter
    if filter != None:
        y = np.where(y == filter)[0]
        x = x[y]
    # reshape it into 28x28 pixels
    x = x.reshape(x.shape[0], 28, 28)
    # calculate the grid layout
    cells = x.shape[0]
    if cells > max_length: cells = max_length
    if int(cells / 10) == cells / 10:
        rows = int(cells / 10)
    else:
        rows = int(cells / 10) + 1
    plt.figure(figsize=(10, rows))
    gs1 = gridspec.GridSpec(rows, 10)
    gs1.update(wspace=0.001, hspace=0.001)  # set the spacing between axes.
    # for each image, plot it into the grid as a subplot
    for i in range(0, cells):
        ax1 = plt.subplot(gs1[i])
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        # if there is a filter, put the index number in red.  Otherwise, put the digit label in blue
        if filter == None:
            label = y[i]
            plt.text(1, 5, label, color='red')
        else:
            if show_index:
                label = y[i]
                plt.text(1, 5, label, color='blue')
        plt.imshow(x[i], cmap=plt.get_cmap('gray_r'), vmax=255)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    if save:
        s = f'.//input//digits//digit_{str(filter)}.png'
        plt.savefig(s)
    if show:
        plt.show()

def main():
    #l oad the mnist data.  In this example we already have it stored in .npy's
    # data must be sent to the plotting function as shape(rows, 784), datatype=uint8, unnormalized(pixel range from 0 to 255)
    xfile = './/input//x_train.npy'
    yfile = './/input//y_train.npy'
    x_train = np.load(file=xfile)
    y_train = np.load(file=yfile)
    perfile = x_train.shape[0]
    #perfile=100  #set this to a small number to test. It takes a LONG time to process an entire mnist dataset
    logmsg('Generating a separate .png for each digit in x_train.')
    for i in range(0, 10):
        logmsg(str(i) + '...')
        plot_mnist_digits_packed(x_train, y_train, max_length=perfile, filter=i, show=False, save=True, show_index=True)
    logmsg('Done.')

if __name__ == "__main__":
    main()
