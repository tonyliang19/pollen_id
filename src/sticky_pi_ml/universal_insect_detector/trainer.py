from detectron2.engine import DefaultTrainer as DefaultDetectronTrainer
from detectron2.data import  build_detection_test_loader, build_detection_train_loader
from sticky_pi_ml.trainer import BaseTrainer
from sticky_pi_ml.universal_insect_detector.ml_bundle import MLBundle


class DetectronTrainer(DefaultDetectronTrainer):
    def __init__(self, ml_bundle: MLBundle):
        self._ml_bundle = ml_bundle
        super().__init__(self._ml_bundle.config)

    def build_train_loader(self, cfg):
        return build_detection_train_loader(self._ml_bundle.config,
                                            mapper=self._ml_bundle.dataset.mapper(self._ml_bundle.config,))

    # def build_test_loader(self, cfg, subdataset_name):
    #     return build_detection_test_loa   der(self._ml_bundle.config, subdataset_name,
    #                                        mapper=self._ml_bundle.dataset.mapper())



class Trainer(BaseTrainer):

    def __init__(self, ml_bundle: MLBundle):
        super().__init__(ml_bundle)
        self._detectron_trainer = DetectronTrainer(ml_bundle)

    def resume_or_load(self, resume: bool = True):
        return self._detectron_trainer.resume_or_load(resume=resume)

    def train(self):
        return self._detectron_trainer.train()



