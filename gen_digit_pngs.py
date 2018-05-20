import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import numpy as np
import datetime
import matplotlib
import pandas as pd
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train_file_csv = ".//input//train.csv"
test_file_csv = ".//input//test.csv"
x_train_file_npy = ".//input//x_train.npy"
y_train_file_npy = ".//input//y_train.npy"
x_test_file_npy = ".//input//x_test.npy"

# fuction that logs to the screen with a timestamp
def logmsg(msg=''):
    fmt = "%H:%M:%S"
    s = f'{datetime.datetime.today().strftime(fmt)}: {msg}'
    print(s)


def convert_csv_to_pnp():
    mnist_train_dataset = pd.read_csv(train_file_csv, delimiter=',').values
    y_train = mnist_train_dataset[:, 0]
    y_train = y_train.astype('uint8')
    x_train = mnist_train_dataset[0:, 1:]
    x_train = x_train.astype('uint8')
    mnist_test_dataset = pd.read_csv(test_file_csv, delimiter=',').values
    x_test = mnist_test_dataset
    x_test = x_test.astype('uint8')
    np.save(x_train_file_npy, x_train)
    np.save(y_train_file_npy, y_train)
    np.save(x_test_file_npy, x_test)

def display_mnist_digit(images, labels, num):
    image = images[num].reshape([28, 28])
    label = labels[num]
    # plt.title('Sample: %d  Label: %d' % (num, label))
    # plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def plot_mnist_digits_packed(x, y, title='title', max_length=100, filter=None, show=False, save=True, show_index=False):
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
        plt.imshow(x[i], cmap=plt.get_cmap('gray_r'), vmax=255)
        # if there is a filter, put the index number in red.  Otherwise, put the digit label in blue
        if y is not None:
            if filter == None:
                label = y[i]
                plt.text(1, 5, label, color='red')
            else:
                if show_index:
                    label = y[i]
                    plt.text(1, 5, label, color='green')
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    if save:
        s = f'.//input//digits//{title}.png'
        plt.savefig(s)
    if show:
        plt.show()

def gen_train_pngs():
    #load the mnist data.  In this example we already have it stored in .npy's
    # data must be sent to the plotting function as shape(rows, 784), datatype=uint8, unnormalized(pixel range from 0 to 255)
    xfile = './/input//x_train.npy'
    yfile = './/input//y_train.npy'
    x = np.load(file=xfile)
    y = np.load(file=yfile)
    perfile = x.shape[0]
    # perfile=100  #set this to a small number to test. It takes a LONG time to process an entire mnist dataset
    logmsg('Generating a separate .png for each digit in x_train.')
    for i in range(0, 10):
        logmsg(str(i) + '...')
        plot_mnist_digits_packed(x, y, title=f'train_{i}', max_length=perfile, filter=i, show=False, save=True, show_index=True)
    logmsg('Done.')

def gen_test_pngs():
    #load the mnist data.  In this example we already have it stored in .npy's
    # data must be sent to the plotting function as shape(rows, 784), datatype=uint8, unnormalized(pixel range from 0 to 255)
    xfile = './/input//x_test.npy'
    x = np.load(file=xfile)
    y = None
    perfile = x.shape[0]
    perfile=4000  #set this to a small number to test. It takes a LONG time to process an entire mnist dataset
    logmsg('Generating a separate .png for each digit in x_test.')
    for i in range(0, x.shape[0], perfile):
        logmsg(str(i) + '...')
        plot_mnist_digits_packed(x[i:i + perfile], y, title=f'test_{i}', max_length=perfile, filter=None, show=False, save=True, show_index=False)
    logmsg('Done.')

def display_single_digit(index):
    #load the mnist data.  In this example we already have it stored in .npy's
    # data must be sent to the plotting function as shape(rows, 784), datatype=uint8, unnormalized(pixel range from 0 to 255)
    xfile = './/input//x_train.npy'
    yfile = './/input//y_train.npy'
    x = np.load(file=xfile)
    y = np.load(file=yfile)

    image = x[index].reshape([28, 28])
    label = y[index]
    plt.title('Sample: %d  Label: %d' % (index, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


def main():
    #convert_csv_to_npy     #we'll convert the csv's to numpy arrays for better data loading speed.  Csv's are slow.
    # gen_train_pngs()
    # gen_test_pngs()
    display_single_digit(35094)



if __name__ == "__main__":
    main()

#0's - 12817
#1's - 60,191,2284,2316,5275,7389,19633,19979,24891,29296,32565,38191,38544,40339,41739
#2's - 4677,7527,9162,13471,16598,20891,27364
#3's - 240,11593,11896,17966,25708,28560,33198,34477,36018,41492
#4's - 1383,6781,22478,23604,26171,26182,26411,18593,34862,36051,36241,36830,37544
#5's - 456,2867,2872,5695,6697,9195,18319,19364,27034,29253,35620,
#6's - 7610,12388,12560,14659,15219,18283,24122,31649,40214,40358,40653,
#7's - 6295,7396,15284,19880,20089,21423,25233,26366,26932,27422,31741,
#8's - 8566,10920,23489,25069,28003,28851,30352,30362,35396,36984,39990,40675,40868,41229,
#9's - 631,4226,9943,14914,15065,17300,18316,19399,20003,20018,23135,23732,29524,33641,40881, 41354