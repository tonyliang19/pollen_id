import unittest
from sticky_pi_ml.image import Image, SVGImage, ImageJsonAnnotations
import os
import logging
import pytz
import datetime
import numpy as np
import tempfile
from sticky_pi_ml.annotations import Annotation
import glob
import shutil

#
# class TestImageSequence(unittest.TestCase):
#     _test_seq = [i for i in sorted(glob.glob('sticky_pi_dir/raw_images/7168f343/*.jpg'))]
#
#     def test_to_video(self):
#         images = [Image(i) for i in self._test_seq]
#         seq = ImageSequence(images)
#
#         target = tempfile.mktemp(prefix='sticky_pi_video_', suffix='.mp4')
#         try:
#             seq.to_animation(target, scale=.25)
#             self.assertTrue(os.path.exists(target))
#
#         finally:
#             import logging
#             # logging.warning(target)
#             os.remove(target)



class TestImage(unittest.TestCase):
    # _test_image = "raw_images/1b74105a/1b74105a"
    _test_image = "raw_images/1b74105a/1b74105a.2020-07-05_10-07-16.jpg"
    # _test_svg_images = ("sticky_pi_dir/raw_images/01abcabc.2020-01-01_01-02-03.svg",
    #                     "sticky_pi_dir/raw_images//d59ff54a.2020-03-03_18-04-20.svg",
    #                     "sticky_pi_dir/raw_images/abcdef00.2021-01-01_00-00-01.svg")
    #
    # _test_legacy_svg_image = "sticky_pi_dir/raw_images/legacy_3b8bb733.2019-12-10_15-46-49.svg"
    # _test_image_nonext = "sticky_pi_dir/raw_images/7168f343/7168f3a3.2019-12-03_23-06-09.jpg"
    # _test_combine_path_svg_image = "sticky_pi_dir/raw_images/combined-path_afbd6f68.2020-04-23_20-49-21.svg"
    _image_shape = (1944, 2592, 3)
    # maxDiff = None

    def test_init(self):
        timezone = pytz.timezone("UTC")
        full_path = os.path.join(os.path.dirname(__file__), self._test_image)
        im = Image(full_path)
        self.assertEqual(im.datetime, datetime.datetime(2020, 7, 5, 10, 7, 16))
        self.assertEqual(im.device, '1b74105a')
        self.assertEqual(im.filename, '1b74105a.2020-07-05_10-07-16.jpg')
        self.assertEqual(im.path, full_path)

    def test_read(self):
        full_path = os.path.join(os.path.dirname(__file__), self._test_image)
        im = Image(full_path)
        array = im.read()
        self.assertEqual(array.shape, self._image_shape)

    def test_to_svg(self):
        self._to_svg(self._test_image)

    #
    # def test_json_image(self):
    #     im = self._test_svg_images[2]
    #     svg_ori = SVGImage(im)
    #     json_annots = svg_ori.json_annotations()
    #     svg_ori.tag_detector_version('hello', 'world')
    #     json_annots = svg_ori.json_annotations()
    #
    #     # ImageJsonAnnotations(json_annots)
    #
    #
    # def test_svg_to_svg(self):
    #     im = self._test_svg_images[2]
    #     svg_ori = SVGImage(im)
    #     array = svg_ori.read()
    #
    #     tmp_dir = tempfile.mkdtemp(prefix='sticky_pi_test_')
    #     try:
    #
    #         target = os.path.join(tmp_dir, '01abcabc.2020-01-01_01-02-03.svg')
    #         svg_ori.to_svg(target)
    #         svg_new = SVGImage(target)
    #         array_new = svg_new.read()
    #         self.assertEqual(np.sum(array - array_new), 0)
    #     finally:
    #         shutil.rmtree(tmp_dir)
    #         pass
    #
    def _to_svg(self, img):

        full_path = os.path.join(os.path.dirname(__file__), img)
        im = Image(full_path)

        contours = [np.array([[[551, 552],[883, 884],[995, 596]]]).transpose((1, 0, 2)),
                    np.array([[[51, 52], [93, 94], [500, 6]]]).transpose((1, 0, 2)),
                    np.array([[[101, 102], [203, 204], [405, 306]]]).transpose((1, 0, 2))]
        an = [Annotation(c, '#ffff00') for c in contours ]

        im.set_annotations(an)

        tmp_dir = tempfile.mkdtemp(prefix='sticky_pi_test_')
        try:

            target = os.path.join(tmp_dir, '01abcabc.2020-01-01_01-02-03.svg')

            im.to_svg(target)
            imsvg = SVGImage(target)

            self.assertEqual(im.shape, imsvg.shape)

            self.assertDictEqual(im.metadata, imsvg.metadata)
            svg_contour = [a._contour for a in imsvg._annotations]
            for c,s in zip(contours, svg_contour):
                self.assertTrue(np.sum(c-s) == 0)

            self.assertTrue(os.path.exists(target))


        finally:
            # logging.warning(tmp_dir)
            shutil.rmtree(tmp_dir)
            pass

    #
    # def test_extract_jpeg(self):
    #     for im in self._test_svg_images:
    #         full_path = os.path.join(os.path.dirname(__file__), im)
    #         svgim = SVGImage(full_path)
    #
    #         tmp_dir = tempfile.mkdtemp(prefix='sticky_pi_test_')
    #         try:
    #             target = os.path.join(tmp_dir, '01abcabc.2020-01-01_01-02-03.jpg')
    #             svgim.extract_jpeg(target)
    #             im = Image(target)
    #             self.assertTrue(np.sum(im.read() - svgim.read()) == 0)
    #
    #         finally:
    #             shutil.rmtree(tmp_dir)
    #             pass
    #
    # def test_to_png(self):
    #
    #     full_path = os.path.join(os.path.dirname(__file__), self._test_image)
    #     im = Image(full_path)
    #
    #     contours = [np.array([[[551, 552],[883, 884],[995, 596]]]).transpose((1, 0, 2)),
    #                np.array([[[51, 52], [93, 94], [500, 6]]]).transpose((1, 0, 2)),
    #                np.array([[[101, 102], [203, 204], [405, 306]]]).transpose((1, 0, 2))]
    #     an = [Annotation(c, '#ffff00') for c in contours ]
    #
    #     im.set_annotations(an)
    #
    #     tmp_dir = tempfile.mkdtemp(prefix='sticky_pi_test_')
    #     try:
    #
    #         target = os.path.join(tmp_dir, '01abcabc.2020-01-01_01-02-03.png')
    #         im.to_png(target)
    #
    #     finally:
    #         import shutil
    #         shutil.rmtree(tmp_dir)
    #
    #
    # def test_legacy_svg(self):
    #
    #     full_path = os.path.join(os.path.dirname(__file__), self._test_legacy_svg_image)
    #     # todo expects warning here "Cannot extract metadata"
    #     svgim = SVGImage(full_path)
    #
    #     tmp_dir = tempfile.mkdtemp(prefix='sticky_pi_test_')
    #     try:
    #
    #         target = os.path.join(tmp_dir, '01abcabc.2020-01-01_01-02-03.jpg')
    #         svgim.extract_jpeg(target)
    #         im = Image(target)
    #         self.assertTrue(np.sum(im.read() - svgim.read()) == 0)
    #
    #     finally:
    #         shutil.rmtree(tmp_dir)
    #         pass
    #
    #
    # def test_combined_svg_path(self):
    #
    #     full_path = os.path.join(os.path.dirname(__file__), self._test_combine_path_svg_image)
    #     # todo expects warning here "Cannot extract metadata"
    #     svgim = SVGImage(full_path)




        #
        # tmp_dir = tempfile.mkdtemp(prefix='sticky_pi_test_')
        # try:
        #
        #     target = os.path.join(tmp_dir, '01abcabc.2020-01-01_01-02-03.jpg')
        #     svgim.extract_jpeg(target)
        #     im = Image(target)
        #     self.assertTrue(np.sum(im.read() - svgim.read()) == 0)
        #
        # finally:
        #     shutil.rmtree(tmp_dir)
        #     pass
