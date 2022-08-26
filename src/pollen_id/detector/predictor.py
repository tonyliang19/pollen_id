import torch
import numpy as np
from shapely.geometry import Polygon
from pollen_id.image import Image
from typing import Tuple, Dict, List, Any
from pollen_id.annotations import Annotation
from pollen_id.predictor import BasePredictor
from pollen_id.detector.ml_bundle import MLBundle
from detectron2.engine import DefaultPredictor
import cv2
import logging
import math


class Predictor(BasePredictor):
    _minimum_tile_overlap = 500
    _score_threshold = 0.85
    _iou_threshold = 0.33

    def __init__(self, ml_bundle: MLBundle):
        super().__init__(ml_bundle)
        self._min_width = self._ml_bundle.config.MIN_MAX_OBJ_SIZE[0]
        self._detectron_predictor = DefaultPredictor(self._ml_bundle.config)

    
    def detect(self, image: Image, *args, **kwargs) -> Image:

        instances = self._detect_instances(image, *args, **kwargs)
        new_image = image.copy()
        new_image.set_annotations(instances)
        new_image.tag_detector_version(self._name, self.version)
        return new_image

    def _mask_to_polygons(self, mask, offset, dilate_kernel_size=3):
        mask = np.ascontiguousarray(mask.cpu())  # some versions of cv2 does not support incontiguous arr
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size))
        mask = cv2.dilate(mask.astype(np.uint8), kernel)
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE,
                                       offset=offset)[-2:]  # for cv3 cv4 compat

        new_contours = []
        for c in contours:
            try:
                rect = cv2.minAreaRect(c)
                (_, _), (width, height), _ = rect

            except Exception as e:
                width, height = 0, 0
                logging.warning(e)

            if width < height:
                width, height = height, width

            if width > self._min_width / 2:
                new_contours.append(c)

        logging.debug("Size-excluded %i contours" % (len(contours) - len(new_contours)))
        contours, new_contours = new_contours, None
        out = []
        for c in contours:
            epsilon = 0.006 * cv2.arcLength(c, True)
            out.append(cv2.approxPolyDP(c, epsilon, True))
        contours = out
        if len(contours) == 0:
            return None
        largest_contour = np.argmax([cv2.contourArea(c) for c in contours])
        return contours[largest_contour]

    def _detect_instances(self, img: Image):
        polys = []
        classes = []
        # logging.debug(img)

        array = img.read()

        # todo
        # make exception to removing edge objects on the edge of the actual image
        # think about what to do when object fully overlap as they come from multiple detections
        # self intersecting contours :(

        if array.shape[1] <= 1024:
            x_range = [0]
            x_n_tiles = 1
        # else:
        #     x_n_tiles = math.ceil(1 + (array.shape[1] - 1024) / (1024 - self._minimum_tile_overlap))
        #     x_stride = (array.shape[1] - 1024) // (x_n_tiles - 1)
        #     x_range = [r for r in range(0, array.shape[1] - 1023, x_stride)]

        if array.shape[0] <= 1024:
            y_range = [0]
            y_n_tiles = 1

        # else:
        #     y_n_tiles = math.ceil(1 + (array.shape[0] - 1024) / (1024 - self._minimum_tile_overlap))
        #     y_stride = (array.shape[0] - 1024) // (y_n_tiles - 1)
        #     y_range = [r for r in range(0, array.shape[0] - 1023, y_stride)]
        offsets = []
        for n, j in enumerate(y_range):
            for m, i in enumerate(x_range):
                offsets.append(((m, n), (i, j)))

        for i, ((m, n), o) in enumerate(offsets):
            logging.info(f"{img.filename}, {i}/{len(offsets)}")
            im_1 = array[o[1]: (o[1] + 1024), o[0]: (o[0] + 1024)]
            p = self._detectron_predictor(im_1)
            p_bt = p['instances'].pred_boxes.tensor
            big_enough = torch.zeros_like(p_bt[:, 0], dtype=torch.bool)
            big_enough = big_enough.__or__(p_bt[:, 2] - p_bt[:, 0] > self._ml_bundle.config.MIN_MAX_OBJ_SIZE[0])
            big_enough = big_enough.__or__(p_bt[:, 3] - p_bt[:, 1] > self._ml_bundle.config.MIN_MAX_OBJ_SIZE[0])

            # we remove redundant edge instances as they should overlap
            non_edge_cases = torch.ones_like(p_bt[:, 0], dtype=torch.bool)

            if m > 0:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 0] > 32)
            if m < x_n_tiles - 1:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 2] < 1024 - 32)

            if n > 0:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 1] > 32)
            if n < y_n_tiles - 1:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 3] < 1024 - 32)


            p['instances'] = p['instances'][non_edge_cases.__and__(big_enough)]
            p['instances'] = p['instances'][p['instances'].scores > self._score_threshold]
            classes_for_one_inst = []
            poly_for_one_inst = []

            #fixme, this could be parallelised
            for i in range(len(p['instances'])):
                instance_offset = (o[0] - self._ml_bundle.config.ORIGINAL_IMAGE_PADDING,
                                   o[1] - self._ml_bundle.config.ORIGINAL_IMAGE_PADDING)

                poly = self._mask_to_polygons(p['instances'].pred_masks[i, :, :],
                                              offset=instance_offset)
                if poly is not None:
                    poly_for_one_inst.append(poly)
                    classes_for_one_inst.append(
                        int(p['instances'].pred_classes[i]) + 1)  # as we want one-indexed classes
            polys.append(poly_for_one_inst)
            classes.append(classes_for_one_inst)

        overlappers = []
        for i in range(len(offsets)):
            overlappers_sub = []
            for j in range(len(offsets)):
                if i != j and abs(offsets[j][1][0] - offsets[i][1][0]) < 1024 and abs(
                        offsets[j][1][1] - offsets[i][1][1]) < 1024:
                    overlappers_sub.append(j)
            overlappers.append(overlappers_sub)

        all_valid = []  # (origin, pred_class, poly)

        # merge predictions from several overlaping tiles
        for origin, poly_one_pred in enumerate(polys):
            for i, p1 in enumerate(poly_one_pred):
                add = True
                p_shape_1 = Polygon(np.squeeze(p1))
                for v in all_valid:
                    if origin not in overlappers[v[0]]:
                        continue
                    p2 = v[2]
                    p_shape_2 = Polygon(np.squeeze(p2))

                    # todo check bounding box overlap
                    try:
                        iou = p_shape_1.intersection(p_shape_2).area / p_shape_1.union(p_shape_2).area
                    except Exception as e:
                        iou = 1  # fixme topological exception
                    if iou > self._iou_threshold:
                        add = False
                        continue
                if add:
                    all_valid.append((origin, classes[origin][i], p1))
        annotation_list = []
        for _, pred_class, poly in all_valid:
            stroke = "#00ff80"
            a = Annotation(poly, parent_image=img, stroke_colour=stroke)
            annotation_list.append(a)
        return annotation_list