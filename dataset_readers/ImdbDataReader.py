from typing import Dict, Optional, Iterable, List

import os.path as osp
import re
from pathlib import Path
import tarfile
from itertools import chain

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
from datasets import load_dataset
import pdb

@DatasetReader.register('imdb')
class ImdbDataReader(DatasetReader):
    TAR_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    TRAIN_DIR = 'aclImdb/train'
    TEST_DIR = 'aclImdb/test'

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                **kwargs):
        super().__init__(**kwargs)

        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        if file_path == 'train':
            temp_dataset = load_dataset('imdb', split='train').shuffle(seed=42)
            dataset = temp_dataset.train_test_split(test_size=0.2, shuffle=False)['train']
        elif file_path == 'test':
            imdb = load_dataset('imdb')
            dataset = imdb['test']
        elif file_path == 'dev':
            temp_dataset = load_dataset('imdb', split='train').shuffle(seed=42)
            dataset = temp_dataset.train_test_split(test_size=0.2, shuffle=False)['test']
        else:
            raise ValueError(f"only 'train', 'dev', and 'test' are valid for 'file_path', but '{file_path}' is given.")

        for _, item in enumerate(dataset):
            yield self.text_to_instance(item['text'], item['label'])

    @overrides
    def text_to_instance(self, source_sequence: str, target: int) -> Instance:
        REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        source_sequence = REPLACE_NO_SPACE.sub("", source_sequence)
        source_sequence = REPLACE_WITH_SPACE.sub(" ", source_sequence)
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(source_sequence)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        fields['label'] = LabelField(target, skip_indexing=True)
        return Instance(fields)

# reader = ImdbDataReader()
# dataset = list(reader.read('dev'))
# print("type of its first element: ", type(dataset[0]))
# print("size of dataset: ", len(dataset))