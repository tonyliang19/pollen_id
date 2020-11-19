import shutil
import tempfile
import torch

class BaseDataset(object):
    _sub_datasets = {'train', 'val'}
    _md5_max_training = 'bf'  # use the md5 of object to allocate to train vs val (md5 > _md5_max_training <=> val)

    def __init__(self, data_dir: str, config, cache_dir: str):
        self._data_dir = data_dir
        self._config = config
        self._cache_dir = cache_dir
        self._is_prepared = False
        self._training_data = []
        self._val_data = []

    def _prepare(self):
        raise NotImplementedError()

    def prepare(self):
        self._prepare()
        assert len(self._training_data) > 0, "Should have at least 1 images in Training set"
        assert len(self._val_data) > 0, "Should have at least 1 images in Validation set"
        self._is_prepared = True

    def get_loader(self, dataset: str):
        raise NotImplementedError()

    def data_loader(self, sub_dataset: str) -> torch.utils.data.DataLoader:
        raise NotImplementedError()

    def visualise(self, subset='train', augment=False):
        raise NotImplementedError()
