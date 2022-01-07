from os import makedirs
from pathlib import Path


class RepositoryInfo:
    def __init__(self, sub_folder_save):
        self.path_tmp = Path(__file__).parent / 'tmp'
        self.path_save = self.path_tmp / sub_folder_save

        makedirs(self.path_tmp, exist_ok=True)
        makedirs(self.path_save, exist_ok=True)
