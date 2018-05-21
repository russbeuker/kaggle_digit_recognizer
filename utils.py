import datetime
import os
import random as rn
from pathlib import Path

import globals


def logmsg(msg=''):
    fmt = "%H:%M:%S"
    s = f'{datetime.datetime.today().strftime(fmt)}: {msg}'
    if globals.log_mode == 'file_only' or globals.log_mode == 'both':
        with open(globals.log_file, "a") as myfile:
            myfile.write(f'{s}\n')
    if globals.log_mode == 'screen_only' or globals.log_mode == 'both':
        print(s)


def delete_log_file():
    my_file = Path(globals.log_file)
    if my_file.is_file():
        os.remove(globals.log_file)


# returns a random "one in val" result.  So for a 1 in 5 chance, pass val=5.
def flip(val):
    return (rn.randint(1, val) == val)
