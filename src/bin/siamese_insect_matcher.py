import os

from sticky_pi_ml.utils import MLScriptParser
from sticky_pi_ml.siamese_insect_matcher.ml_bundle import ClientMLBundle
from sticky_pi_api.client import LocalClient #, RemoteClient
from sticky_pi_ml.siamese_insect_matcher.trainer import Trainer
from sticky_pi_ml.siamese_insect_matcher.predictor import Predictor

BUNDLE_NAME = 'siamese-insect-matcher'

if __name__ == '__main__':
    parser = MLScriptParser()
    option_dict = parser.get_opt_dict()

    bundle_dir = os.path.join(option_dict['BUNDLE_ROOT_DIR'], BUNDLE_NAME)
    ml_bundle_cache = os.path.join(bundle_dir,'.cache')
    os.makedirs(ml_bundle_cache, exist_ok=True)


    if option_dict['action'] == 'fetch':
        #todo use remote client here
        client = LocalClient(option_dict['LOCAL_CLIENT_DIR'])
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'])
        ml_bundle.sync_remote_to_local()

    elif option_dict['action'] == 'qc':
        raise NotImplementedError

    elif option_dict['action'] == 'eval':
        raise NotImplementedError

    elif option_dict['action'] == 'train':
        client = LocalClient(option_dict['LOCAL_CLIENT_DIR'])
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        t = Trainer(ml_bundle)
        t.resume_or_load(resume=not option_dict['restart_training'])
        t.train()

    elif option_dict['action'] == 'predict':
        client = LocalClient(option_dict['LOCAL_CLIENT_DIR'])
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'], cache_dir=ml_bundle_cache)
        pred = Predictor(ml_bundle)
        pred.detect_client()


    elif option_dict['action'] == 'push':
        #todo use remote client here
        client = LocalClient(option_dict['LOCAL_CLIENT_DIR'])
        ml_bundle = ClientMLBundle(bundle_dir, client, device=option_dict['device'])
        ml_bundle.sync_local_to_remote()
