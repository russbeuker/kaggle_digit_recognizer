import os
from datetime import datetime
from os import rename
from shutil import rmtree
from uuid import uuid4


class TrainingSession():
    def __init__(self, name=None, parent_dir=None, log_mode='screen', timestamped_folder=False, delete_folder=True):
        self.name = name
        self.parent_dir = parent_dir
        self.delete_folder = delete_folder
        self.timestamped_folder = timestamped_folder

        if self.timestamped_folder:
            fmt = "%H-%M-%S"
            s = f'{datetime.today().strftime(fmt)}'
            self.full_path = self.parent_dir + self.name + '-' + s + '//'
        else:
            self.full_path = self.parent_dir + self.name + '//'

        self.log_path = self.full_path + 'log.txt'
        self.state_path = self.full_path + 'state.txt'
        self.results_path = self.full_path + 'results.txt'
        self.log_mode = log_mode
        self.create_dir()
        self.log('Session created.')

    def create_dir(self):
        if self.delete_folder:
            if os.path.exists(self.full_path):
                tmp_name = self.parent_dir + str(uuid4())
                rename(self.full_path, tmp_name)
                rmtree(tmp_name)
            os.makedirs(self.full_path)
        else:
            if not os.path.exists(self.full_path):
                os.makedirs(self.full_path)

    def log(self, msg=''):
        fmt = "%H:%M:%S"
        s = f'{datetime.today().strftime(fmt)}: {msg}'
        if self.log_mode == 'file_only' or self.log_mode == 'both':
            with open(self.log_path, "a") as myfile:
                myfile.write(f'{s}\n')
        if self.log_mode == 'screen_only' or self.log_mode == 'both':
            print(s)
