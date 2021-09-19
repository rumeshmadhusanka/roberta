from typing import Dict, Optional, Iterable, List

import os.path as osp
import re
from pathlib import Path
import tarfile
from itertools import chain
import json

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
import pdb

@DatasetReader.register('math')
class MathT5DataReader(DatasetReader):
    TAR_URL = 'https://people.eecs.berkeley.edu/~hendrycks/MATH.tar'
    TRAIN_DIR = 'MATH/train'
    # DEV_DIR = 'RACE/dev/high'
    TEST_DIR = 'MATH/test'

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 max_tokens: int = None,
                 source_prefix: Optional[str] = "math: ",
                **kwargs):
        super().__init__(**kwargs)

        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.source_prefix = source_prefix

    @overrides
    def _read(self, file_path):
        """
        Data processing method was originally from: https://arxiv.org/pdf/2005.00700.pdf
        """
        tar_path = cached_path(self.TAR_URL)
        tf = tarfile.open(tar_path, 'r')
        cache_dir = Path(osp.dirname(tar_path))
        if not (cache_dir / self.TRAIN_DIR).exists() and not (cache_dir / self.TEST_DIR).exists():
            tf.extractall(cache_dir)

        if file_path == 'train':
            cache_dir = osp.join(cache_dir, self.TRAIN_DIR)
        elif file_path == 'test':
            cache_dir = osp.join(cache_dir, self.TEST_DIR)
        else:
            raise ValueError(f"only 'train', 'dev', and 'test' are valid for 'file_path', but '{file_path}' is given.")
        # path = chain(Path(cache_dir.joinpath(pos_dir)).glob('*.txt'),
        #              Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))
        for type in ["intermediate_algebra", "geometry", "algebra", "precalculus", "prealgebra", "counting_and_probability", "number_theory"]:
            path = Path(osp.join(cache_dir, type)).glob('*.json')

            for p in path:
                data = json.loads(p.read_text())
                problem = data["problem"]
                solution = data["solution"]
                yield self.text_to_instance(self.source_prefix + problem, solution)

    @overrides
    def text_to_instance(self, source_sequence: str, target_sequence: str) -> Instance:
        # REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        # REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        # source_sequence = REPLACE_NO_SPACE.sub("", source_sequence)
        # source_sequence = REPLACE_WITH_SPACE.sub(" ", source_sequence)
        fields: Dict[str, Field] = {}
        if self.source_prefix is not None:
            tokens = self._tokenizer.tokenize(self.source_prefix + source_sequence)
        else:
            tokens = self._tokenizer.tokenize(source_sequence)
        target = self._tokenizer.tokenize(target_sequence)
        fields['source_tokens'] = TextField(tokens, self._token_indexers)
        fields['target_tokens'] = TextField(target, self._token_indexers)
        return Instance(fields)

# reader = MathT5DataReader()
# dataset = list(reader.read('test'))
# print("type of its first element: ", type(dataset[0]))
# print("size of dataset: ", len(dataset))