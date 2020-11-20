import os
import dotenv
import argparse
import logging
import hashlib


def md5(file, chunksize=32768):
    # if the file is a path, open and recurse
    if type(file) == str:
        with open(file, 'rb') as f:
            return md5(f)
    try:
        hash_md5 = hashlib.md5()
        for chunk in iter(lambda: file.read(chunksize), b""):
            hash_md5.update(chunk)
    finally:
        file.seek(0)
    return hash_md5.hexdigest()


class MLScriptParser(argparse.ArgumentParser):
    _valid_actions = {'fetch', 'train', 'qc', 'eval', 'push'}
    _required_env_vars = ['BUNDLE_ROOT_DIR', 'LOCAL_CLIENT_DIR']

    def __init__(self, config_file=None):
        super().__init__()

        self.add_argument("action", help=str(self._valid_actions))

        self.add_argument("-v", "--verbose", dest="verbose", default=False,
                          help="verbose",
                          action="store_true")

        self.add_argument("-D", "--debug", dest="debug", default=False,
                          help="debug",
                          action="store_true")

        self.add_argument("-r", "--restart-training", dest="restart_training", default=False, action="store_true")
        self.add_argument("-g", "--gpu", dest="gpu", default=False, help="GPU", action="store_true")
        self._config_file = config_file

    def _get_env_conf(self):
        if self._config_file is not None:
            assert os.path.isfile(self._config_file)
            dotenv.load_dotenv(self._config_file)

        out = {}
        for var_name in self._required_env_vars:
            out[var_name] = os.getenv(var_name)
            if not out[var_name]:
                raise ValueError('No environment variable `%s''' % var_name)
        return out

    def get_opt_dict(self):
        args = self.parse_args()

        option_dict = vars(args)
        if option_dict['action'] not in self._valid_actions:
            logging.error('Wrong action!')
            self.print_help()
            exit(1)

        if option_dict['gpu']:
            option_dict['device'] = 'cuda'
        else:
            option_dict['device'] = 'cpu'

        if option_dict['verbose']:
            logging.getLogger().setLevel(logging.INFO)

        if option_dict['debug']:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.info("DEBUG mode ON")

        # 'BUNDLE_DIR', 'LOCAL_CLIENT_DIR'
        env_conf = self._get_env_conf()
        option_dict.update(env_conf)
        return option_dict

#