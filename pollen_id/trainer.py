from abc import ABC

from pollen_id.ml_bundle import BaseMLBundle
from pollen_id.predictor import BasePredictor


class BaseTrainer(ABC):
    def __init__(self, ml_bundle: BaseMLBundle):
        self._ml_bundle = ml_bundle
        self._ml_bundle.dataset.prepare()

    def train(self):
        raise NotImplementedError()

    def resume_or_load(self, resume: bool = True):
        raise NotImplementedError()

    def validate(self, predictor: BasePredictor, out_dir: str = None):
        raise NotImplementedError()
