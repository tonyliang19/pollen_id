import math
import cv2
import pickle
import gzip

import itertools
from typing import List
import glob
import os
import copy
import numpy as np
import torch


from multiprocessing import Pool
from functools import partial

import logging

from torchvision.transforms import ColorJitter, ToTensor

from detectron2.data import detection_utils
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation import Augmentation
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from sticky_pi_ml.dataset import BaseDataset
from sticky_pi_ml.image import SVGImage
from sticky_pi_ml.utils import md5
from sticky_pi_ml.universal_insect_detector.palette import Palette


def _objs_from_svg(svg_path, config, palette):
    min_size, max_size = config.MIN_MAX_OBJ_SIZE
    svg_img = SVGImage(svg_path)
    try:
        out = []
        for a in svg_img.annotations:
            width = a.rot_rect_width()
            if width <= min_size or width > max_size:
                continue
            seg = [a.contour.flatten().astype(float).tolist()]
            try:
                label_id = palette.get_id_annot(a)
            except Exception as e:
                logging.warning(svg_img.filename + str(e))
                continue
            obj = {
                "bbox": a.bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": seg,
                "category_id": label_id - 1,
                "iscrowd": 0
            }
            out.append(obj)

    except Exception as e:
        logging.error("issue reading %s" % svg_img.filename)
        raise e
    return out


def _pickled_objs_from_svg(file, cache_dir, palette, config):
    basename = os.path.basename(file)
    pre, ext = os.path.splitext(basename)
    new_basename = pre + '.mask.pgz'
    new_path = os.path.join(cache_dir, new_basename)

    if os.path.exists(new_path):
        with gzip.GzipFile(new_path, 'r') as f:
            out = pickle.load(f)
        return out

    to_pickle = _objs_from_svg(file, config, palette)
    with gzip.GzipFile(new_path, 'w') as f:
        pickle.dump(to_pickle, f)
    return _pickled_objs_from_svg(file, cache_dir, palette, config)


def _create_jpg_from_svg(file, cache_dir):
    basename = os.path.basename(file)
    pre, ext = os.path.splitext(basename)
    new_basename = pre + '.jpg'
    new_path = os.path.join(cache_dir, new_basename)
    if not os.path.exists(new_path):
        SVGImage(file, foreign=True, skip_annotations=True).extract_jpeg(new_path)
    return new_path


def _parse_one_image(svg_file, cache_dir, palette, config):
    pre_extracted_jpg = _create_jpg_from_svg(svg_file, cache_dir)

    with open(pre_extracted_jpg, 'rb') as im_file:
        md5_sum = md5(im_file)
    # todo file can be a MEMORY BUFFER
    h, w, _ = cv2.imread(pre_extracted_jpg).shape

    im_dic = {'file_name': pre_extracted_jpg,
              'height': h,
              'width': w,
              'image_id': os.path.basename(pre_extracted_jpg),
              'annotations': _pickled_objs_from_svg(svg_file, cache_dir, palette, config),
              "md5": md5_sum,
              "original_svg": svg_file
              }

    return im_dic


class OurColorJitter(Augmentation):

    def __init__(self, brightness, contrast, saturation, hue):
        self._tv_transform = ColorJitter(brightness, contrast, saturation, hue)

        super().__init__()
        # self._init(locals())

    def get_transform(self, image):
        with torch.no_grad():
            img = torch.from_numpy(image.transpose((2, 0, 1))).contiguous()
            # img = torch.zeros_like(img)
            image = self._tv_transform.forward(img)
            image = image.numpy().transpose((1, 2, 0))
        return T.BlendTransform(src_image=image, src_weight=1, dst_weight=0)


class DatasetMapper(object):

    def __init__(self, cfg, augment=True):
        # fixme add these augmentations in config ?
        self._augment = augment
        self.tfm_gens = [
            T.RandomRotation(angle=[0, 360], sample_style='range', expand=False),
            T.RandomCrop(crop_type='absolute', crop_size=cfg.INPUT.CROP.SIZE),
            OurColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
            T.RandomFlip(horizontal=True, vertical=False),
            T.RandomFlip(horizontal=False, vertical=True),
        ]


        self._padding = cfg.ORIGINAL_IMAGE_PADDING
        self.img_format = cfg.INPUT.FORMAT


    def __call__(self, dataset_dict):


        # we padd the image to make a sementic difference between real edges and stitching edges
        if not self._augment:
            return self._validation_crops(dataset_dict)
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = detection_utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image = cv2.copyMakeBorder(image, self._padding, self._padding,
                                   self._padding, self._padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        for obj in dataset_dict["annotations"]:
            bbox = (obj["bbox"][0] + self._padding, obj["bbox"][1] + self._padding, obj["bbox"][2], obj["bbox"][3])
            obj["bbox"] = bbox

            obj['segmentation'] = np.add(obj['segmentation'], self._padding).tolist()

        annots = [
            detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        instances = detection_utils.annotations_to_instances(annots, image.shape[:2])

        dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

        return dataset_dict

    def _validation_crops(self, dataset_dict):
        image = detection_utils.read_image(dataset_dict["file_name"], format=self.img_format)
        INPUT_SIZE = 1024
        h, w, _ = image.shape
        n_img_columns = 1 + (w // INPUT_SIZE)
        y_padding = math.ceil((INPUT_SIZE - (w % INPUT_SIZE)) / 2)
        y_padding_2 = math.floor((INPUT_SIZE - (w % INPUT_SIZE)) / 2)

        n_img_rows = 1 + (h // INPUT_SIZE)
        x_padding = math.ceil((INPUT_SIZE - (h % INPUT_SIZE)) / 2)
        x_padding_2 = math.floor((INPUT_SIZE - (h % INPUT_SIZE)) / 2)

        # print("h, w, x_padding, y_padding")
        # print(h, w, x_padding, y_padding)

        image = cv2.copyMakeBorder(image, x_padding, x_padding_2,
                                   y_padding, y_padding_2, cv2.BORDER_CONSTANT, value=(0, 0, 0))



        out = []

        for i,j in itertools.product(range(n_img_rows), range(n_img_columns)):
            # sub_image = image[i * INPUT_SIZE: (i+1) * INPUT_SIZE, j * INPUT_SIZE: (j+1) * INPUT_SIZE, :]
            local_dataset_dict = copy.deepcopy(dataset_dict)
            tr = [T.CropTransform(j * INPUT_SIZE, i * INPUT_SIZE, INPUT_SIZE, INPUT_SIZE)]
            sub_image, transforms = T.apply_transform_gens(tr, image)
            local_dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

            for obj in local_dataset_dict["annotations"]:
                bbox = (obj["bbox"][0] + self._padding, obj["bbox"][1] + self._padding, obj["bbox"][2], obj["bbox"][3])
                obj["bbox"] = bbox

                obj['segmentation'] = np.add(obj['segmentation'], self._padding).tolist()

            annots = [
                detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in local_dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            instances = detection_utils.annotations_to_instances(annots, image.shape[:2])

            detection_utils.filter_empty_instances(instances)
            local_dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
            out.append(local_dataset_dict)
        return out

class Dataset(BaseDataset):
    def __init__(self, data_dir, config, cache_dir):
        super().__init__(data_dir, config, cache_dir)
        self._palette = None

    def _prepare(self):
        self._palette = Palette({k: v for k, v in self._config.CLASSES})
        # for d in self._sub_datasets:
        #     sub_ds_name = self._name + '_' + d
        input_img_list = sorted(glob.glob(os.path.join(self._data_dir, '*.svg')))
        # assert len(input_img_list) > 1, "Should have at least 2 svg images in %s. Just got %i" % \
        #                                 (self._data_dir, len(input_img_list))
        data = self._serialise_imgs_to_dicts(input_img_list)

        while len(data) > 0:
            entry = data.pop()
            if entry['md5'] > self._md5_max_training:
                self._validation_data.append(entry)
            else:
                self._training_data.append(entry)


        DatasetCatalog.register(self._config.DATASETS.TRAIN[0], lambda : self._training_data)
        MetadataCatalog.get(self._config.DATASETS.TRAIN[0]).set(thing_classes=self._config.CLASSES)

        DatasetCatalog.register(self._config.DATASETS.TEST[0], lambda : self._validation_data)
        MetadataCatalog.get(self._config.DATASETS.TEST[0]).set(thing_classes=self._config.CLASSES)

        logging.info(f"N_train = {len(self._training_data)}: {[i['file_name'] for i in DatasetCatalog.get(self._config.DATASETS.TEST[0])]}")
        logging.info(f"N_validation = {len(self._validation_data)}: {[i['file_name'] for i in DatasetCatalog.get(self._config.DATASETS.TRAIN[0])]}")




    def _serialise_imgs_to_dicts(self, input_img_list: List[str]):

        with Pool(self._config.DATALOADER.NUM_WORKERS) as p:
            out = [m for m in p.map(
                partial(_parse_one_image, cache_dir=self._cache_dir, palette=self._palette, config=self._config),
                input_img_list)]

        return out

    def visualise(self, subset='train', augment=False):
        from detectron2.utils.visualizer import Visualizer
        self.prepare()
        if subset == 'train':
            subset = self._config.DATASETS.TRAIN[0]
        elif subset == 'val':
            subset = self._config.DATASETS.TEST[0]
        else:
            raise ValueError('Unexpected subset. must be train or val')

        tl = build_detection_train_loader(self._config, mapper=DatasetMapper(self._config))
        metadata = MetadataCatalog.get(subset)
        scale = 1
        for batch in tl:
            for per_image in batch:
                img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                # img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                target_fields = per_image["instances"].get_fields()
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                )
                cv2.imshow('training_data', vis.get_image()[:, :, ::-1])
                if cv2.waitKey(-1) == 27:
                    return None

    def mapper(self, config, augment=True):
        return DatasetMapper(config, augment)

    # not used
    def _get_torch_data_loader(self):
        raise NotImplementedError()
