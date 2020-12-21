import gzip
import os
import shutil
import urllib.request

from gensim.models import KeyedVectors
from tqdm import tqdm

from textattack.shared.utils.install import path_in_cache

from gensim.models.keyedvectors import Word2VecKeyedVectors


def read_config():
    config_path = os.path.join(os.path.abspath(__file__), 'config.yaml')
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def provide_word2vec(embedding_name='cc.en.300.vec.gz'):
    return Word2VecKeyedVectors.load(provide_word2vec_file(embedding_name))


def provide_word2vec_file(embedding_name):
    # TODO: generalize to support fasttext OR word2vec
    def download_word2vec():
        """Download, extract and pickle word2vec embedding. clean up afterwards.

        Args:
            embedding_name: name of embedding (e.g. 'cc.en.300.vec') to download from fastext website (https://fasttext.cc).

        Returns: None
        """

        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        def download_to(url, output_path):
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

        def download():
            print(f'Downloading embedding and saving as {embedding_path_gz} ...', file=sys.stderr)
            download_to(embedding_url, embedding_path_gz)

        def extract():
            print(f'Extracting embedding to {embedding_path_vec} ...', file=sys.stderr)
            with gzip.open(embedding_path_gz, 'rb') as f_in:
                with open(embedding_path_vec, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        def pickle():
            print(f'Saving final embedding as {embedding_path_m} ...', file=sys.stderr)
            embedding = KeyedVectors.load_word2vec_format(embedding_path_vec)
            embedding.save(embedding_path_m)

        def clean_up():
            tmp_folder = os.path.join(download_path, 'tmp')
            for filename in os.listdir(tmp_folder):
                file_path = os.path.join(tmp_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        if not os.path.exists(embedding_path_m):
            download()
            extract()
            pickle()
            clean_up()

    embedding_url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{embedding_name}.gz'
    download_path = path_in_cache('word2vec')
    embedding_path_gz = os.path.join(download_path, 'tmp', embedding_name)
    embedding_path_vec = os.path.splitext(embedding_path_gz)[0]
    embedding_path_m = os.path.join(download_path, os.path.basename(os.path.splitext(embedding_path_vec)[0] + '.m'))

    if not os.path.exists(embedding_path_m):
        download_word2vec()

    return embedding_path_m
