import json
import os
import logging
import numpy as np
import cv2
import torch

from shapely.geometry import Polygon

from detectron2.engine import DefaultTrainer as DefaultDetectronTrainer, PeriodicWriter
from detectron2.data import  build_detection_test_loader, build_detection_train_loader
from detectron2.engine import HookBase
import detectron2.utils.comm as comm


from sticky_pi_ml.universal_insect_detector.predictor import Predictor
from sticky_pi_ml.image import SVGImage
from sticky_pi_ml.utils import iou, iou_match_pairs
from sticky_pi_ml.trainer import BaseTrainer
from sticky_pi_ml.universal_insect_detector.ml_bundle import MLBundle
from sticky_pi_ml.universal_insect_detector.palette import Palette


# def custom_test_mapper(dataset_dict):
#     # it will be modified by code below
#     dataset_dict = copy.deepcopy(dataset_dict)
#     image = utils.read_image(dataset_dict["file_name"], format="BGR")
#     transform_list = []
#     instances = utils.annotations_to_instances(annos, image.shape[:2])
#     dataset_dict["instances"] = utils.filter_empty_instances(instances)
#     return dataset_dict
#
#
# def build_test_loader(cls, cfg, dataset_name="my_dataset_val"):
#     return build_detection_test_loader(cfg, dataset_name, mapper=get_custom_test_mapper())
#


class ValLossHook(HookBase):

    def __init__(self, cfg, dataset_name):
        super().__init__()
        self.cfg = cfg.clone()
        self._dataset_name = dataset_name
        self._last_validation_output = None

    def after_step(self):
        """
            After each step calculates the validation loss and adds it to the train storage
        """
        if self._last_validation_output is None or ( self.trainer.iter + 1) % self.cfg.TEST_PERIOD == 0:

            loader = self.trainer.build_test_loader(self.cfg, self._dataset_name)

            with torch.no_grad():
                all_losses = []
                overall_n_instances = 0
                for d in loader:

                    loss_dict = self.trainer.model(d)
                    losses = sum(loss_dict.values())
                    assert torch.isfinite(losses).all(), loss_dict

                    loss_dict_reduced = {"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                    total_loss = sum(loss for loss in loss_dict_reduced.values())
                    loss_dict_reduced["val_total_loss"] = total_loss
                    n_instances = sum([len(i["instances"]) for i in d])
                    # all_losses_times_n_instances.append(loss_dict_reduced/n_instances)
                    loss_dict_reduced["val_instance_weighted_total_loss"] = total_loss * n_instances
                    all_losses.append(loss_dict_reduced)
                    overall_n_instances += n_instances
                    
                out = None
                for d in all_losses:
                    if out is None:
                        out = {k: v / len(all_losses) for k, v in d.items()}
                    else:
                        for k, v in d.items():
                            out[k] += v / len(all_losses)
                out["val_instance_weighted_total_loss"] /= overall_n_instances
                out["val_instance_weighted_total_loss"] *= len(all_losses)
                self._last_validation_output = out

                if comm.is_main_process():
                    self.trainer.storage.put_scalars(**self._last_validation_output)


class DetectronTrainer(DefaultDetectronTrainer):
    def __init__(self, ml_bundle: MLBundle):
        self._ml_bundle = ml_bundle
        super().__init__(self._ml_bundle.config)

    def build_train_loader(self, cfg):
        return build_detection_train_loader(self._ml_bundle.config,
                                            mapper=self._ml_bundle.dataset.mapper(self._ml_bundle.config))

    def build_test_loader(self, cfg, dataset_name):

        return build_detection_test_loader(self._ml_bundle.config, dataset_name,
                                           mapper=self._ml_bundle.dataset.mapper(self._ml_bundle.config, augment=False))


class Trainer(BaseTrainer):
    def __init__(self, ml_bundle: MLBundle):
        super().__init__(ml_bundle)
        self._detectron_trainer = DetectronTrainer(ml_bundle)

        self._detectron_trainer.register_hooks(
            [ValLossHook(self._ml_bundle.config, self._ml_bundle.name + "_val")]
        )

        for hook in self._detectron_trainer._hooks:
            if isinstance(hook, PeriodicWriter):
                hook._period = self._ml_bundle.config.PRINTING_PERIOD

        periodic_writer_hook = [hook for hook in self._detectron_trainer._hooks if isinstance(hook, PeriodicWriter)]

        all_other_hooks = [hook for hook in self._detectron_trainer._hooks if not isinstance(hook, PeriodicWriter)]
        self._detectron_trainer._hooks = all_other_hooks + periodic_writer_hook

    def resume_or_load(self, resume: bool = True):
        return self._detectron_trainer.resume_or_load(resume=resume)

    def train(self):
        return self._detectron_trainer.train()

    def validate(self, predictor: Predictor, out_dir: str = None, gt_fill_colour: str = '#0000ff'):
        out = []
        for v in self._ml_bundle.dataset.validation_data:
            original_svg = v["original_svg"]

            gt = SVGImage(original_svg)
            im = predictor.detect(gt)
            o = self._score_vs_gt(gt, im)

            if out_dir is not None:
                assert os.path.isdir(out_dir), 'out_dir must be an existing directory'
                target = os.path.join(out_dir, "validation_" + gt.filename)
                logging.info('saving %s' % target)
                new_annot = []

                for gta in gt.annotations:
                    gta._fill_colour = gt_fill_colour
                    gta._fill_opacity = .3
                    new_annot.append(gta)
                for a in im.annotations:
                    new_annot.append(a)
                im.set_annotations(new_annot)
                im.to_svg(target)
            out.extend(o)
        if out_dir is not None:
            with open(os.path.join(out_dir, 'results.json'), 'w') as f:
                json.dump(out, f)
        return out

    def _score_vs_gt(self, gt, im,  iou_threshold=0.33):
        out = []
        palette = Palette({k: v for k, v in self._ml_bundle.config.CLASSES})
        for c in palette.classes:
            c_stroke = palette.get_stroke_from_class(c)
            gt_annot = [a for a in gt.annotations if a.stroke_col == c_stroke]
            im_annot = [a for a in im.annotations if a.stroke_col == c_stroke]
            s = self._score_vs_gt_one_class(gt_annot, im_annot, iou_threshold, c, gt.filename)
            out.extend(s)
        return out

    def _score_vs_gt_one_class(self, gt_a, im_a, iou_threshold, obj_class, filename):
        out = []

        # if len(gt_a) == 0:
            # if len(im_a) == 0:
            #     return {'recall': 1,
            #             'precision': 1}
            # else:
            #     return {'recall': 0,
            #             'precision': 0}
        #     for
        # if len(im_a) == 0:
        #
        #     return {'recall': 0,
        #             'precision': 0}

        if len(im_a) == 0:
            if len(gt_a) == 0:
                return {}
            pairs = [(i,None) for i, _  in enumerate(gt_a)]

        elif len(gt_a) == 0:
            pairs = [(None, i) for i, _ in enumerate(im_a)]

        else:
            arr = np.zeros((len(gt_a), len(im_a)), dtype=np.float)
            for m, g in enumerate(gt_a):
                g_shape = Polygon(np.squeeze(g.contour))
                for n, i in enumerate(im_a):
                    i_shape = Polygon(np.squeeze(i.contour))
                    # todo check bounding box overlap
                    iou_val = iou(g_shape, i_shape)
                    arr[m, n] = iou_val
            pairs = iou_match_pairs(arr, iou_threshold)

        for g_a, i_a in pairs:
            if i_a:
                in_im = True
            else:
                in_im = False

            if g_a is not None:
                area = cv2.contourArea(gt_a[g_a].contour)
                in_gt = True

            else:
                in_gt = False
                area = cv2.contourArea(im_a[i_a].contour)

            out.append({'area': area,
                        'in_gt': in_gt,
                        'in_im': in_im,
                        'class': obj_class,
                        'filename': filename})
        return out



