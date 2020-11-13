from sticky_pi_ml.ml_bundle import BaseMLBundle


class BaseTrainer(object):
    def __init__(self, ml_bundle: BaseMLBundle):
        self._ml_bundle = ml_bundle
        self._ml_bundle.dataset.prepare()

    def train(self):
        raise NotImplementedError()

    def resume_or_load(self, resume: bool = True):
        raise NotImplementedError()
