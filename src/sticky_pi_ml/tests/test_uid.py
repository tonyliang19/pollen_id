import glob
import os
import logging
import shutil
import tempfile
import unittest
from sticky_pi_ml.universal_insect_detector.ml_bundle import MLBundle, ClientMLBundle
from sticky_pi_ml.universal_insect_detector.trainer import Trainer
from sticky_pi_ml.universal_insect_detector.predictor import Predictor
from sticky_pi_api.client import LocalClient

logging.getLogger().setLevel(logging.INFO)


class TestMLBundle(unittest.TestCase):
    _bundle_dir = './uid_bundle'
    _test_image = "raw_images/5c173ff2/5c173ff2.2020-06-20_21-33-24.jpg"

    def test_MLBundle(self):
        bndl = MLBundle(self._bundle_dir)
    #
    def test_ClientMLBundle(self):
        client_temp_dir = tempfile.mkdtemp(prefix='sticky_pi_client_')

        # the di dirname is used to identify the ML bundle
        temp_dst_bundle = os.path.join(tempfile.mkdtemp(prefix='sticky_pi_test_'), 'uid_bundle')
        os.makedirs(temp_dst_bundle)
        try:
            cli = LocalClient(client_temp_dir)

            bndl = ClientMLBundle(self._bundle_dir, cli)
            bndl.sync_local_to_remote()

            # should warn the ML bundle is empty at this stage
            bndl2 = ClientMLBundle(temp_dst_bundle, cli)

            # should reinit the bundle after dl the data to the local dir
            bndl2.sync_remote_to_local()
            # bndl2 = ClientMLBundle(bndl2._root_dir, cli)

            cf1 = bndl._config
            cf2 = bndl2._config
            cf1.MODEL.WEIGHTS = None
            cf2.MODEL.WEIGHTS = None
            cf1.OUTPUT_DIR = None
            cf2.OUTPUT_DIR = None

            self.assertDictEqual(cf1, cf2)
            self.assertEqual([os.path.basename(p) for p in sorted(glob.glob(os.path.join(bndl.dataset._data_dir)))],
                             [os.path.basename(p) for p in sorted(glob.glob(os.path.join(bndl2.dataset._data_dir)))])



        finally:
            shutil.rmtree(client_temp_dir)
            shutil.rmtree(temp_dst_bundle)
            pass

    def test_Trainer(self):
        bndl = MLBundle(self._bundle_dir)
        # bndl.dataset.visualise(augment=True)
        t = Trainer(bndl)
    #     t.resume_or_load(resume=True)

        # t.train()
    def test_Predictor(self):
        bndl = MLBundle(self._bundle_dir)
        pred = Predictor(bndl)
    #     im = Image(self._test_image)
    #     pred.detect(im)


