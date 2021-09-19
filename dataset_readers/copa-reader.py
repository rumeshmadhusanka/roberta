import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register('copa-reader')
class ImdbDataReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 **kwargs):
        super().__init__(**kwargs)

        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        url = "https://people.ict.usc.edu/~gordon/downloads/COPA-resources.tgz"
        directory = Path(cached_path(url, extract_archive=True))
        if file_path == "dev":
            data = os.path.join(directory, "COPA-resources/datasets", "copa-dev.xml")
        elif file_path == "test":
            data = os.path.join(directory, "COPA-resources/datasets", "copa-test.xml")
        elif file_path == "all":
            data = os.path.join(directory, "COPA-resources/datasets", "copa-all.xml")
        else:
            raise ValueError(f"only 'all', 'dev', and 'test' are valid for 'file_path', but '{file_path}' is given.")
        tree = ET.parse(data)
        for child in tree.getroot():
            p = child.findtext("p")
            a1 = child.findtext("a1")
            a2 = child.findtext("a2")
            id_at = child.attrib['id']
            asks = child.attrib['asks-for']
            alt = child.attrib['most-plausible-alternative']
            # yield the necessary values
            yield None

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


reader = ImdbDataReader()
dataset = reader._read("train")
next(dataset)
# print("type of its first element: ", type(dataset[0]))
# print("size of dataset: ", len(dataset))
