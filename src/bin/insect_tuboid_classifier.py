import logging
import os
from sticky_pi_ml.utils import MLScriptParser
from sticky_pi_api.client import LocalClient #, RemoteClient
from sticky_pi_ml.insect_tuboid_classifier.ml_bundle import ClientMLBundle
from sticky_pi_ml.insect_tuboid_classifier.trainer import Trainer
from sticky_pi_ml.insect_tuboid_classifier.predictor import Predictor


BUNDLE_NAME = 'insect-tuboid-classifier'
VALIDATION_OUT_DIR = 'validation_results'

if __name__ == '__main__':
    parser = MLScriptParser()
    option_dict = parser.get_opt_dict()
    bundle_dir = os.path.join(option_dict['BUNDLE_ROOT_DIR'], BUNDLE_NAME)
    ml_bundle_cache = os.path.join(bundle_dir, '.cache')
    os.makedirs(ml_bundle_cache, exist_ok=True)


    if option_dict['action'] == 'fetch':
        #todo use remote client here
        client = LocalClient(option_dict['LOCAL_CLIENT_DIR'])
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'])
        ml_bundle.sync_remote_to_local()

    elif option_dict['action'] == 'qc':
        raise NotImplementedError

    elif option_dict['action'] == 'validate':
        client = LocalClient(option_dict['LOCAL_CLIENT_DIR'])
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        t = Trainer(ml_bundle)
        predictor = Predictor(ml_bundle)
        t.validate(predictor, VALIDATION_OUT_DIR)

    elif option_dict['action'] == 'train':
        client = LocalClient(option_dict['LOCAL_CLIENT_DIR'])
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        t = Trainer(ml_bundle)
        t.resume_or_load(resume=not option_dict['restart_training'])
        t.train()

    elif option_dict['action'] == 'predict':
        client = LocalClient(option_dict['LOCAL_CLIENT_DIR'])
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        predictor = Predictor(ml_bundle)
        predictor.predict_client(device="%", start_datetime="2020-01-01_00-00-00", end_datetime="2100-01-01_00-00-00")


    elif option_dict['action'] == 'push':
        #todo use remote client here
        client = LocalClient(option_dict['LOCAL_CLIENT_DIR'])
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'])
        ml_bundle.sync_local_to_remote()
