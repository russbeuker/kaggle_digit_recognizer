from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import pandas as pd
from pathlib import Path

# get the most recent state of the test
my_file = Path('.//sessions//test1//results - Copy.txt')
if my_file.is_file():
    df = pd.read_csv(my_file)
    batchsizes = df['BATCHSIZE'].tolist()
    dropouts = df['DROPOUT'].tolist()
    vals = df['AVG_METRIC_VAL'].tolist()
    print(df.head())

    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(batchsizes, dropouts, vals)
    pyplot.show()

else:
    quit()


