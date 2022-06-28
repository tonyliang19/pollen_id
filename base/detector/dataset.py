# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# import image as img
import torch
# from PIL import Image as im
import numpy as np
import cv2
from multiprocessing import Pool
import os
import gzip
# import itertools
import copy
import glob
# import math
import logging
import pickle
from typing import List
from base.utils import md5
from base.dataset import BaseDataset
from functools import partial
from sticky_pi_ml.image import SVGImage
from detectron2.structures.boxes import BoxMode
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data import transforms as T
from detectron2.data import detection_utils as du
from detectron2.data import build_detection_test_loader, build_detection_train_loader


try:
    from detectron2.data.transforms.augmentation import Augmentation
    from detectron2.data.transforms import CropTransform
except ImportError:
    from detectron2.data.transforms.transform_gen import TransformGen as Augmentation

    class CropTransform(Augmentation):

        def __init__(self, x0: int, y0: int, w: int, h: int):
            super().__init__()
            self._init(locals())

        def get_transform(self, img):
            return T.CropTransform(self.x0, self.y0, self.w, self.h)


class Dataset(BaseDataset):
    def __init__(self, data_dir, cache_dir, config):
        super().__init__(data_dir=data_dir, cache_dir=cache_dir, config=config)
        # print(f"this is __init_, cache dir = {self._cache_dir}, config = {self._config}")

    def _prepare(self):
        input_img_list = sorted(glob.glob(os.path.join(self._data_dir, "*.svg")))
        data = self._serialise_imgs_to_dicts(input_img_list)
        # print(f"length of data: {len(data)}")
        while len(data) > 0:
            entry = data.pop()
            # print(entry["file_name"])

            if entry["md5"] > self._md5_max_training:
                # for dat in self._validation_data:
                #     if entry["file_name"] not in dat["file_name"]:
                # for e in self._validation_sub_image(entry):
                #     print(f"Appending to validation")
                self._validation_data.append(entry)

            else:
                self._training_data.append(entry)

        DatasetCatalog.register(self._config.DATASETS.TRAIN[0], lambda: self._training_data)
        MetadataCatalog.get(self._config.DATASETS.TRAIN[0]).set(thing_classes=self._config.CLASSES)
        DatasetCatalog.register(self._config.DATASETS.TEST[0], lambda: self._validation_data)
        MetadataCatalog.get(self._config.DATASETS.TEST[0]).set(thing_classes=self._config.CLASSES)

        logging.info(
            f"N_train = {len(self._training_data)}")
        logging.info(
            f"N_validation = {len(self._validation_data)}")

    def _serialise_imgs_to_dicts(self, input_img_list: List[str]):

        # 2 is number of workers, change to variable from config instead
        # with Pool(self._config.DATALOADER.NUM_WORKERS) as p:
        #     out = [m for m in p.map(
        #         partial(_parse_one_image, cache_dir=self._cache_dir,
        #                 config=self._config),
        #         input_img_list)]
        out = []
        for m in input_img_list:
            out.append(_parse_one_image(m, cache_dir=self._cache_dir,
                        config=self._config))
        ## TODO
        ## ^^
        ## problem is with parsing svg image
        return out

    def visualise(self, subset="train", augment=False):
        if not self._is_prepared:
            self.prepare()
        if subset == "train":
            subset = self._config.DATASETS.TRAIN[0]
            t1 = build_detection_train_loader(self._config,
                                              mapper=DatasetMapper(self._config))

        elif subset == "val":
            subset = self._config.DATASETS.TEST[0]
            t1 = build_detection_test_loader(self._config,
                                             self._config.DATASETS.TEST[0],
                                             mapper=DatasetMapper(self._config,
                                                                  augment=False))
        else:
            raise ValueError('Unexpected subset. must be train or val')

        metadata = MetadataCatalog.get(subset)
        scale = 1
        for batch in t1:
            for per_image in batch:
                img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                target_fields = per_image["instances"].get_fields()
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                )
                cv2.imshow(subset, vis.get_image()[:, :, ::-1])
                if cv2.waitKey(30) == 27:
                    return None


    def mapper(self, config, augment=True):
        return DatasetMapper(config, augment)


class DatasetMapper(object):
    def __init__(self, cfg, augment=True):
        self._augment = augment
        # add more transformations vvv here below
        self.tfm_gens = [
            T.RandomRotation(angle=[0, 360], sample_style="range", expand=False),
            T.RandomFlip(horizontal=True, vertical=False)
        ]

        # self._padding = cfg.ORIGINAL_IMAGE_PADDING
        self._padding = 10
        self.img_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        if not self._augment:
            # print(dataset_dict)
            return self._validation_crops(dataset_dict) ## CHECK HERE <----

        image = du.read_image(dataset_dict["file_name"], format=self.img_format)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annots = []
        for obj in dataset_dict.pop("annotations"):
            if obj.get("iscrowd", 0) == 0:
                try:
                    source_obj = copy.deepcopy(obj)
                    ann = du.transform_instance_annotations(obj, transforms,
                                                            image.shape[:2])
                    annots.append(ann)

                except Exception as e:
                    logging.error(f"Annotation error in {dataset_dict['file_name']}: {source_obj}")
                    logging.error(e)

        instances = du.annotations_to_instances(annots, image.shape[:2])

        dataset_dict["instances"] = du.filter_empty_instances(instances)
        return dataset_dict

    def _validation_crops(self, dataset_dict):
        image = du.read_image(dataset_dict["file_name"], format=self.img_format)

    #     #y_pad = dataset_dict["padding"]["top"]
    #     #x_pad = dataset_dict["padding"]["left"]
    #     tr = [CropTransform(**dataset_dict["cropping"])]
        # tr = []

    #     # for obj in dataset_dict["annotations"]:
    #     #     bbox = (obj["bbox"][0] + x_pad, obj["bbox"][1] + y_pad, obj["bbox"][2], obj["bbox"][3])
    #     #     obj["bbox"] = bbox

    #     #     a = np.array(obj['segmentation'])
    #     #     a[0, 0::2] +=  x_pad
    #     #     a[0, 1::2] +=  y_pad
    #     #     obj['segmentation'] = a.tolist()

        #image, transforms = T.apply_transform_gens(tr, image)

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32")).contiguous()
#         annots = [
#            #du.transform_instance_annotations(obj, transforms, image.shape[:2])
#            for obj in dataset_dict.pop("annotations")
#            #if obj.get("iscrowd", 0) == 0
# ]
        annots = dataset_dict["annotations"]
        #print(f"annots: {annots}")
        instances = du.annotations_to_instances(annots, image.shape[:2])
        dataset_dict["instances"] = du.filter_empty_instances(instances)
        #print(f"Instances: {dataset_dict['instances']}")
        return dataset_dict
    
    
def _objs_from_svg(svg_path, config):
    min_size, max_size = config.MIN_MAX_OBJ_SIZE
    svg_img = SVGImage(svg_path, foreign=True)
    excluded = 0
    try:
        out = []
        for a in svg_img.annotations:
            width = a.rot_rect_width()
            if width <= min_size or width > max_size:
                excluded += 1
                continue
            seg = [a.contour.flatten().astype(float).tolist()]
            obj = {
                "bbox": a.bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": seg,
                "category_id": 0,
                "iscrowd": 0
            }
            out.append(obj)
    except Exception as e:
        logging.error("issue reading %s" % svg_img.filename)
        raise e
    if excluded:
        logging.info(f"Excluded {excluded} contours in {svg_img.filename} due to size")
    return out


def _pickled_objs_from_svg(file, cache_dir, config):
    basename = os.path.basename(file)
    pre, ext = os.path.splitext(basename)
    new_basename = pre + ".mask.pgz"
    new_path = os.path.join(cache_dir, new_basename)

    if os.path.exists(new_path):
        with gzip.GzipFile(new_path, "r") as f:
            out = pickle.load(f)
        return out
    to_pickle = _objs_from_svg(file, config)
    with gzip.GzipFile(new_path, "w") as f:
        pickle.dump(to_pickle, f)
    return _pickled_objs_from_svg(file, cache_dir=cache_dir, config=config)


def _create_jpg_from_svg(file, cache_dir):
    basename = os.path.basename(file)
    pre, ext = os.path.splitext(basename)
    new_basename = pre + ".jpg"
    new_path = os.path.join(cache_dir, new_basename)
    if not os.path.exists(new_path):
        SVGImage(file, foreign=True, skip_annotations=True).extract_jpeg(new_path)
    return new_path


def _parse_one_image(svg_file, cache_dir, config):
    # print(f"This is parse one image: , {cache_dir}, {config}")
    # return
    pre_extracted_jpg = _create_jpg_from_svg(svg_file, cache_dir)

    with open(pre_extracted_jpg, 'rb') as im_file:
        md5_sum = md5(im_file)
    # todo file can be a MEMORY BUFFER
    h, w, _ = cv2.imread(pre_extracted_jpg).shape

    im_dic = {'file_name': pre_extracted_jpg,
              'height': h,
              'width': w,
              'image_id': os.path.basename(pre_extracted_jpg),
              'annotations': _pickled_objs_from_svg(svg_file, cache_dir=cache_dir, config=config),
              "md5": md5_sum,
              "original_svg": svg_file
              }

    return im_dic
