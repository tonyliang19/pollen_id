import tempfile
import shutil
import os
import numpy as np
import torch
import logging
from typing import Dict, Tuple, List, Any
import pandas as pd
from sticky_pi_api.types import InfoType
from sticky_pi_ml.annotations import Annotation
from sticky_pi_ml.predictor import BasePredictor
from sticky_pi_ml.insect_tuboid_classifier.model import ResNetPlus, make_resnet
from sticky_pi_ml.tuboid import TiledTuboid
from sticky_pi_ml.insect_tuboid_classifier.ml_bundle import MLBundle, ClientMLBundle
from sticky_pi_ml.insect_tuboid_classifier.dataset import OurTorchDataset


class Predictor(BasePredictor):
    # submit to client N at a time predictions
    _client_predict_chunk_size = 64

    def __init__(self, ml_bundle: MLBundle):
        super().__init__(ml_bundle)
        self._net = self._make_net()
        self._taxonomy_mapper = self._ml_bundle.dataset.taxonomy_mapper
        weights = self._ml_bundle.weight_file
        self._net.load_state_dict(torch.load(weights))
        self._net.eval()

    def _make_net(self):
        return make_resnet(pretrained=False, n_classes=self._ml_bundle.dataset.n_classes)

    def predict_client(self, device, start_datetime, end_datetime):
        assert issubclass(type(self._ml_bundle), ClientMLBundle), \
            "This method only works for MLBundles linked to a client"
        client = self._ml_bundle.client
        logging.info('Processing series %s' % [device, start_datetime, end_datetime])

        series = {'device': device,
                  'start_datetime': start_datetime,
                  'end_datetime': end_datetime}

        client_resp = client.get_tiled_tuboid_series_itc_labels([series], what='data')

        if len(client_resp) < 1:
            logging.warning('No tuboids in %s' % series)
            return

        tiled_tuboids_for_series = pd.DataFrame(client_resp)
        # the fields from  the insect tuboid classifier are suffixed by `_itc`. not to be confused with the fields from
        # the tiled tuboids table
        if 'algo_version_itc' not in tiled_tuboids_for_series.columns:
            logging.info('no labels for this series %s. All %i tuboids will be labeled' %
                         (series, len(tiled_tuboids_for_series)))

            tiled_tuboids_for_series['algo_version_itc'] = None
            tiled_tuboids_for_series['algo_name_itc'] = ""

        # we want to label tuboids that are not labeled yet by this algorithm (name) or this version of the algo
        conditions = (self.version > tiled_tuboids_for_series.algo_version_itc) | \
                     (tiled_tuboids_for_series.algo_version_itc.isnull()) | \
                     (self.name != tiled_tuboids_for_series.algo_name_itc)

        tiled_tuboids_for_series = tiled_tuboids_for_series[conditions]
        tiled_tuboids_for_series = tiled_tuboids_for_series.sort_values(by=['algo_version_itc', 'start_datetime'])
        tiled_tuboids_for_series = tiled_tuboids_for_series.drop_duplicates(subset=['tuboid_id'], keep='last')

        if len(tiled_tuboids_for_series) == 0:
            logging.warning('No tuboids to label in %s (all labeled)' % series)
            return

        all_predictions = []
        for _, r in tiled_tuboids_for_series.iterrows():
            temp_dir = tempfile.mkdtemp()
            try:
                tuboid_dir = os.path.join(temp_dir, r['tuboid_id'])
                os.makedirs(tuboid_dir)
                for f in ['metadata', 'tuboid', 'context']:
                    shutil.copy(r[f], tuboid_dir)

                tiled_tuboid = TiledTuboid(tuboid_dir)
                prediction = self.predict(tiled_tuboid)
            finally:
                shutil.rmtree(temp_dir)

            prediction['algo_version'] = self.version
            prediction['algo_name'] = self.name
            prediction['tuboid_id'] = r['tuboid_id']

            logging.info('Prediction: %s' % prediction)

            all_predictions.append(prediction)
            if len(all_predictions) >= self._client_predict_chunk_size:
                logging.info('Sending batch of predictions through client')
                client.put_itc_labels(all_predictions, what='data')
                all_predictions.clear()

        if len(all_predictions) > 0:
            logging.info('Sending last batch of predictions through client')
            client.put_itc_labels(all_predictions)

    def predict(self, tiled_tuboid: TiledTuboid):
        data_entry = OurTorchDataset.tiled_tuboid_to_dict(tiled_tuboid, unsqueezed=True)
        preds = self._net(data_entry)
        preds = preds.detach().numpy()
        label = int(np.argmax(preds))

        out = self._taxonomy_mapper.label_to_level_dict(label)
        pattern = self._taxonomy_mapper.label_to_pattern(label)
        out.update({'label': label,
                'pattern': pattern,
                })
        return out
