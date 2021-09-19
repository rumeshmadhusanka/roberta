from typing import Dict, List, Optional
import logging

from allennlp.data import Tokenizer
from overrides import overrides
from nltk.tree import Tree

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.common.checks import ConfigurationError

from pathlib import Path
from itertools import chain
import os.path as osp
import zipfile
import re
from tqdm import tqdm as tqdm
import numpy as np
import math
from datasets import load_dataset
import pdb
import json

logger = logging.getLogger(__name__)

TRAIN_VAL_SPLIT_RATIO = 0.9


@DatasetReader.register("wsc")
class WscDatasetReader(DatasetReader):
    TAR_URL = 'https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip'
    DIR = "WSC"

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or \
                {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        tar_path = cached_path(self.TAR_URL)
        tf = zipfile.ZipFile(tar_path, 'r')
        cache_dir = Path(osp.dirname(tar_path))
        if not (cache_dir / self.DIR).exists():
            tf.extractall(cache_dir)

        if file_path == 'train':
            path = str(Path(cache_dir)) + "/WSC/train.jsonl"
        # elif file_path == 'test':
        #     path = str(Path(cache_dir)) + "/WSC/test.jsonl"
        elif file_path == 'dev':
            path = str(Path(cache_dir)) + "/WSC/val.jsonl"
        else:
            raise ValueError(f"only 'train', 'dev', and 'test' are valid for 'file_path', but '{file_path}' is given.")
        with open(path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            item = json.loads(json_str)
            options = item['target']["span1_text"] + " , " + item['target']["span2_text"]
            yield self.text_to_instance(item['text'], options, str(item['label']))

    @overrides
    def text_to_instance(
            self, text: str, options: str, label: str = None) -> Optional[Instance]:
        REPLACE_WITH_SPACE = re.compile("\n\t")
        REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        text = REPLACE_NO_SPACE.sub("", text)
        text = REPLACE_WITH_SPACE.sub(" ", text)
        text_tokens = self._tokenizer.tokenize(text)[:448]
        option_tokens = self._tokenizer.tokenize(options)[:60]
        tokens = self._tokenizer.add_special_tokens(text_tokens, option_tokens)
        print(tokens)
        text_field = TextField(tokens, self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)

# reader = WscDatasetReader()
# dataset = reader._read('dev')
# print(next(dataset))
# pdb.set_trace()
# print("type of its first element: ", type(dataset))
