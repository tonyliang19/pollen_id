import os
from sticky_pi_ml.dataset import BaseDataset


class BaseMLBundle(object):
    _data_dirname = 'data'      #
    _config_dirname = 'config'  #
    _config_filename = 'config.yaml'
    _output_dirname = 'output'   #
    _model_filename = 'model_final.pth'

    _name = None
    _DatasetClass = None  # must be implemented

    def __init__(self, root_dir: str, device: str = 'cpu', cache_dir=None):
        """
        An abstract class that organises all the components of a ML project:

        * training/validation data files
        * configuration files
        * weight files (i.e. pretrained or resulting of training)

        All components are stored in ``root_dir`` and the class provided utilities to parse inputs, generate ``torch``
        datasets, synchronise the data to an API...
        :param root_dir: the location of the files
        """
        self._root_dir = root_dir
        assert os.path.isdir(root_dir), "%s is not a directory" % root_dir
        config_file = os.path.join(self._root_dir, self._config_dirname, self._config_filename)
        assert os.path.isfile(config_file), "Configuration file %s is does not exist" % config_file
        self._output_dir = os.path.join(self._root_dir, self._output_dirname)

        if cache_dir is None:
            import tempfile
            import atexit
            import shutil
            cache_dir = tempfile.mkdtemp(prefix='sticky_pi_%s_' % self._name)
            atexit.register(shutil.rmtree, cache_dir)

        self._cache_dir = cache_dir

        self._data_dir = os.path.join(self._root_dir, self._data_dirname)
        assert os.path.isdir(self._data_dir), "%s is not a directory" % self._data_dir

        self._weight_file = os.path.join(self._output_dir, self._model_filename)
        self._config = self._configure(config_file, device)
        self._dataset = self._DatasetClass(self._data_dir, self._config, self._cache_dir)

    def _configure(self, config_file, device):
        raise NotImplementedError

    @property
    def dataset(self) -> BaseDataset:
        return self._dataset


    @property
    def config(self) -> dict:
        return self._config
