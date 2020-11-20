import os
from sticky_pi_ml.dataset import BaseDataset
from sticky_pi_ml.utils import md5
from sticky_pi_api.client import BaseClient
import logging

class BaseMLBundle(object):
    _data_dirname = 'data'      #
    _config_dirname = 'config'  #
    _config_filename = 'config.yaml'
    _output_dirname = 'output'   #
    _model_filename = 'model_final.pth'

    _name = None
    _DatasetClass = None  # must be implemented

    def __init__(self, root_dir: str, device: str = 'cpu', cache_dir=None):
        """
        An abstract class that organises all the components of a ML project:

        * training/validation data files
        * configuration files
        * weight files (i.e. pretrained or resulting of training)

        All components are stored in ``root_dir`` and the class provided utilities to parse inputs, generate ``torch``
        datasets, synchronise the data to an API...
        :param root_dir: the location of the files
        """
        self._cache_dir = cache_dir
        self._root_dir = root_dir
        self._device = device

        if not os.path.isdir(root_dir):
            logging.warning("%s is not a directory, creating it" % root_dir)
            # assert os.path.dirname(os.path.normpath(root_dir)),
            os.mkdir(root_dir)

        self._output_dir = os.path.join(self._root_dir, self._output_dirname)
        self._config_dir = os.path.join(self._root_dir, self._config_dirname)
        self._data_dir = os.path.join(self._root_dir, self._data_dirname)

        if self._cache_dir is None:
            import tempfile
            import atexit
            import shutil
            self._cache_dir = tempfile.mkdtemp(prefix='sticky_pi_%s_' % self._name)
            atexit.register(shutil.rmtree, self._cache_dir)

            os.makedirs(self._output_dir, exist_ok=True)

        config_file = os.path.join(self._config_dir, self._config_filename)
        self._weight_file = os.path.join(self._output_dir, self._model_filename)

        if not os.path.isdir(self._data_dir):
            logging.warning("Data dir does not exist. making it: %s" % self._data_dir)
            os.makedirs(self._data_dir, exist_ok=True)

        if not os.path.isdir(self._config_dir):
            logging.warning("config dir does not exist. making it: %s" % self._config_dir)
            os.makedirs(self._config_dir, exist_ok=True)

        if not os.path.isdir(self._output_dir):
            logging.warning("Model dir does not exist. making it: %s" % self._output_dir)
            os.makedirs(self._output_dir, exist_ok=True)

        if not os.path.isfile(config_file):
            logging.warning("Configuration file %s is does not exist (yet?)" % config_file)
            self._config = None

        else:
            self._config = self._configure(config_file, self._device)

        self._dataset = self._DatasetClass(self._data_dir, self._config, self._cache_dir)

    def _configure(self, config_file, device):
        raise NotImplementedError

    @property
    def dataset(self) -> BaseDataset:
        return self._dataset

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> dict:
        return self._config

    @property
    def version(self):
        file = self._weight_file
        m = md5(file)
        return "%i-%s" % (os.path.getmtime(file), m)


class BaseClientMLBundle(BaseMLBundle):
    def __init__(self, root_dir: str, client: BaseClient, device: str = 'cpu', cache_dir=None):
        super().__init__(root_dir, device, cache_dir)
        self._client = client

    def sync_local_to_remote(self, what: str = 'all'):
        assert what in {'all', 'data', 'model'}
        self._client.put_ml_bundle_dir(self._name, self._root_dir, what)

    def sync_remote_to_local(self, what: str = 'all'):
        assert what in {'all', 'data', 'model'}
        self._client.get_ml_bundle_dir(self._name, self._root_dir, what)
        self.__init__(self._root_dir, self._client, self._device, self._cache_dir)

        # config_file = os.path.join(self._root_dir, self._config_dirname, self._config_filename)
        # if not os.path.isfile(config_file):
        #     logging.warning("Configuration file %s is does not exist (yet?)" % config_file)
        #
        # self._config = self._configure(config_file, self._device)


# class BaseRemoteMLBundle(BaseMLBundle):
#
#     _multipart_chunk_size = 8 * 1024 * 1024
#     _is_remote = True
#     _allowed_suffixes = ['.yaml', 'model_final.pth', '.svg']
#
#     def __init__(self, root_dir: str, device: str = 'cpu', cred_end_file =None):
#         super().__init__(root_dir, device)
#
#         self._get_credentials(cred_end_file)
#
#         if not os.path.exists(self._root_dir):
#             logging.info("Creating root dir for ml bundle: %s" % self._root_dir)
#             os.makedirs(self._root_dir, exist_ok=True)
#
#         # self._cache_remote_bucket()
#
#     def _get_credentials(self, cred_env_file):
#         if cred_env_file:
#             dotenv.load_dotenv(cred_env_file)
#
#         self._bucket_conf = {
#             "aws_secret_access_key": os.environ.get("S3BUCKET_PRIVATE_KEY"),
#             "aws_access_key_id": os.environ.get("S3BUCKET_ACCESS_KEY"),
#             "bucket": self._name,
#             "S3BUCKET_HOST": os.environ.get("S3BUCKET_HOST")
#         }
#
#         for k, v in self._bucket_conf.items():
#             if not v:
#                 raise Exception("No value for bucket credential `%s`. Check file and/or env variables" % k)
#
#         self._bucket_conf['endpoint_url'] = "http://%s" % self._bucket_conf['S3BUCKET_HOST']
#
#
#     def sync_local_to_remote(self):
#         # typically run when training is finished
#         # send local files based on md5 (keep dir structure)
#         client = boto3.resource('s3', **self._bucket_conf)
#
#         # fixme ensure versioning is enabled
#         # versioning = client.BucketVersioning(self._bucket_conf['bucket'])
#         # versioning.enable()
#
#         bucket = client.Bucket(self._bucket_conf['bucket'])
#         for root, dirs, files in os.walk(self._root_dir, topdown=True, followlinks=True):
#             for name in files:
#
#                 matches = [s for s in self._allowed_suffixes if name.endswith(s)]
#                 if len(matches) == 0:
#                     continue
#
#                 path = os.path.join(root, name)
#                 # if name.endswith('.pth') and path != name.:
#                 #     logging.debug('skipping non-last weight file: %s' % path)
#                 #     continue
#
#                 key = os.path.relpath(path, self._root_dir)
#                 objs = list(bucket.objects.filter(Prefix=key))
#                 objs = [o for o in objs if o.key == key]
#
#                 if len(objs) > 1:
#                     raise Exception("One, and only one, object should match key %s. Several: %s" % (key, str(objs)))
#                 if len(objs) == 0 or self._overwrite_remote_file(path, objs[0]):
#                     logging.info("%s => %s" % (path, key))
#                     bucket.upload_file(path, key)
#
#     def _overwrite_remote_file(self, path, obj):
#         key = obj.key
#         remote_md5 = obj.e_tag[1:-1]
#         remote_last_modified = obj.last_modified
#         with open(path, 'rb') as t:
#             local_md5 = multipart_etag(t, chunk_size=self._multipart_chunk_size)
#         timezone = pytz.timezone('UTC')
#         local_last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(path))
#         local_last_modified = timezone.localize(local_last_modified)
#         if local_md5 == remote_md5:
#             return False
#
#         if local_last_modified > remote_last_modified:
#             return True
#         elif local_last_modified == remote_last_modified:
#             return False
#         else:
#             logging.warning('There is a newer version of file %s on remote. NOT uploading ' % key)
#             return False
#
#     def _overwrite_local_file(self, obj, cache_dir):
#         file = obj.key
#         remote_md5 = obj.e_tag[1:-1]
#         remote_last_modified = obj.last_modified
#         target = os.path.join(cache_dir, file)
#         if not os.path.exists(target):
#             return True
#         with open(target, 'rb') as t:
#             local_md5 = multipart_etag(t, chunk_size=self._multipart_chunk_size)
#
#         timezone = pytz.timezone('UTC')
#         local_last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(target))
#         local_last_modified = timezone.localize(local_last_modified)
#
#         if local_md5 == remote_md5:
#             return False
#
#         if local_last_modified > remote_last_modified:
#             logging.warning('File %s exits both on remote and local, and have different md5s,'
#                             'but it appears newer on local. NOT overwriting!. %s != %s' % (
#                             file, local_md5, remote_md5))
#             return False
#
#         elif local_last_modified == remote_last_modified:
#             return False
#         else:
#             return True
#
#     def sync_remote_to_local(self, model_only=False):
#         client = boto3.resource('s3', **self._bucket_conf)
#
#         # fixme ensure versioning is enabled. now, hangs
#
#         # versioning = client.BucketVersioning(self._bucket_conf['bucket'])
#         # logging.warning("versioning.status() : ")
#         # logging.warning(versioning.status())
#         # versioning.enable()
#
#         bucket = client.Bucket(self._bucket_conf['bucket'])
#
#
#         for obj in bucket.objects.all():
#             dirname = os.path.dirname(obj.key)
#             # just download model without training data
#             if model_only and os.path.basename(obj.key) != self._model_filename:
#                 continue
#             if dirname:
#                 os.makedirs(os.path.join(self._root_dir, dirname), exist_ok=True)
#
#             if self._overwrite_local_file(obj, self._root_dir):
#                 logging.info("%s => %s", obj.key, os.path.join(self._root_dir, obj.key))
#                 bucket.download_file(obj.key, os.path.join(self._root_dir, obj.key))
#                 # here, when we write the local file, its time stamp is updated to the one of the remote!
#                 # otherwise it would be redownloaded redownload it!
#                 filename = os.path.join(self._root_dir, obj.key)
#                 stat = os.stat(filename)
#                 atime = stat.st_atime
#                 os.utime(filename, times=(atime, obj.last_modified.timestamp()))
#             else:
#                 logging.info("Skipping %s", obj.key)
#
#

