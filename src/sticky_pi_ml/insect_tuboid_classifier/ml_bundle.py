
from sticky_pi_ml.ml_bundle import BaseMLBundle, BaseClientMLBundle
from sticky_pi_ml.insect_tuboid_classifier.dataset import Dataset
import yaml


class MLBundle(BaseMLBundle):
    _name = 'insect-tuboid-classifier'
    _DatasetClass = Dataset

    def _configure(self, config_file, device):
        with open(config_file, 'r') as file:
            config = yaml.load(file, Loader=yaml.SafeLoader)

        config['WEIGHTS'] = self._weight_file
        config['DEVICE'] = device
        return config


class ClientMLBundle(MLBundle, BaseClientMLBundle):
    pass
