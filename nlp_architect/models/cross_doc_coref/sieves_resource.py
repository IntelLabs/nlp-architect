# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from nlp_architect import LIBRARY_ROOT
from nlp_architect.data.cdc_resources.relations.relation_types_enums import \
    WikipediaSearchMethod, OnlineOROfflineMethod, EmbeddingMethod


class SievesResources(object):
    def __init__(self):
        """Cross Document co-reference sieve system resources configuration class"""

        self.__eval_output_dir = str(LIBRARY_ROOT / 'datasets' / 'cdc' / 'test_predict')

        self.__elastic_index = 'enwiki_v2'
        self.__elastic_host = 'localhost'
        self.__elastic_port = 9200
        self.__wiki_folder = str(LIBRARY_ROOT / 'dumps'  'wikipedia')
        self.__wd_file = str(LIBRARY_ROOT / 'dump' / 'within_doc_core' / 'ecb_wd_coref_proc.json')
        self.__wn_folder = str(LIBRARY_ROOT / 'dump' / 'wordnet')
        self.__elmo_file = str(LIBRARY_ROOT / 'dump' / 'embedde' / 'ecb_all_with_stop_elmo.pickle')
        self.__glove_file = str(LIBRARY_ROOT / 'dump' / 'embedde' / 'ecb_all_embed_glove.pickle')
        self.__referent_dict_file = str(LIBRARY_ROOT / 'dataset' / 'coref.dict1.tsv')
        self.__vo_dict_file = str(LIBRARY_ROOT / 'dataset' / 'verbocean.unrefined.2004-05-20.txt')

        self.__wiki_search_method = WikipediaSearchMethod.ONLINE
        self.__wn_search_method = OnlineOROfflineMethod.ONLINE
        self.__embed_search_method = EmbeddingMethod.ELMO
        self.__referent_dict_method = OnlineOROfflineMethod.ONLINE
        self.__vo_search_method = OnlineOROfflineMethod.ONLINE

    @property
    def eval_output_dir(self) -> str:
        """
        The output dir of the evaluation files, here scorer file for cross doc coref spans will
        be saved
        """
        return self.__eval_output_dir

    @eval_output_dir.setter
    def eval_output_dir(self, eval_output_dir: str):
        self.__eval_output_dir = eval_output_dir

    @property
    def wiki_folder(self):
        """
        Location of Wikipedia mini data set file, #Required mini data set file location for Offline
        evaluation using Wikipedia sieve
        """
        return self.__wiki_folder

    @wiki_folder.setter
    def wiki_folder(self, wiki_folder: str):
        self.__wiki_folder = wiki_folder

    @property
    def wd_file(self):
        """
        Location of Within doc data set file, #Required when using Within doc sieve
        """
        return self.__wd_file

    @wd_file.setter
    def wd_file(self, wd_file: str):
        self.__wd_file = wd_file

    @property
    def wn_folder(self):
        """
        Location of WordNet mini data set file, #Required mini data set file location for Offline
        evaluation using WordNet sieve
        """
        return self.__wn_folder

    @wn_folder.setter
    def wn_folder(self, wn_folder: str):
        self.__wn_folder = wn_folder

    @property
    def glove_file(self):
        """
        Location of GloVe mini data set file, #Required mini data set file location for Offline
        evaluation using GloVe sieve
        """
        return self.__glove_file

    @glove_file.setter
    def glove_file(self, glove_file: str):
        self.__glove_file = glove_file

    @property
    def elmo_file(self):
        """
        Location of Elmo mini data set file, #Required mini data set file location for Offline
        evaluation using GloVe sieve
        """
        return self.__elmo_file

    @elmo_file.setter
    def elmo_file(self, elmo_file: str):
        self.__elmo_file = elmo_file

    @property
    def referent_dict_file(self):
        """
        Location of Referent dic data set file, #Required mini data set file for Offline
        evaluation or original file for Online evaluation using Referent Dict sieve
        """
        return self.__referent_dict_file

    @referent_dict_file.setter
    def referent_dict_file(self, referent_dict_file: str):
        self.__referent_dict_file = referent_dict_file

    @property
    def vo_dict_file(self):
        """
        Location of VerbOcean data set file, #Required mini data set file for Offline evaluation or
        original file for Online evaluation using VerbOcean sieve
        """
        return self.__vo_dict_file

    @vo_dict_file.setter
    def vo_dict_file(self, vo_dict_file: str):
        self.__vo_dict_file = vo_dict_file

    @property
    def elastic_index(self):
        """
        Elastic index name, #Required when using Elastic evaluation using Wikipedia sieve
        """
        return self.__elastic_index

    @elastic_index.setter
    def elastic_index(self, elastic_index: str):
        self.__elastic_index = elastic_index

    @property
    def elastic_host(self):
        """
        Elastic host, #Required when using Elastic evaluation using Wikipedia sieve
        """
        return self.__elastic_host

    @elastic_host.setter
    def elastic_host(self, elastic_host: str):
        self.__elastic_host = elastic_host

    @property
    def elastic_port(self):
        """
        Elastic port number, #Required when using Elastic evaluation using Wikipedia sieve
        """
        return self.__elastic_port

    @elastic_port.setter
    def elastic_port(self, elastic_port: str):
        self.__elastic_port = elastic_port

    @property
    def wiki_search_method(self):
        """
        Wikipedia search method type, one of: WikipediaSearchMethod.ONLINE,
        WikipediaSearchMethod.OFFLINE, WikipediaSearchMethod.ELASTIC
        """
        return self.__wiki_search_method

    @wiki_search_method.setter
    def wiki_search_method(self, wiki_search_method: str):
        self.__wiki_search_method = wiki_search_method

    @property
    def wn_search_method(self):
        """
        Wordnet search method type, one of: OnlineOROfflineMethod.ONLINE,
        OnlineOROfflineMethod.OFFLINE
        """
        return self.__wn_search_method

    @wn_search_method.setter
    def wn_search_method(self, wn_search_method: str):
        self.__wn_search_method = wn_search_method

    @property
    def embed_search_method(self):
        """
        Wordnet search method type, one of: EmbeddingMethod.GLOVE, EmbeddingMethod.GLOVE_OFFLINE,
        EmbeddingMethod.ELMO, EmbeddingMethod.ELMO_OFFLINE
        """
        return self.__embed_search_method

    @embed_search_method.setter
    def embed_search_method(self, embed_search_method: str):
        self.__embed_search_method = embed_search_method

    @property
    def referent_dict_method(self):
        """
        Referent Dict search method type, one of: OnlineOROfflineMethod.ONLINE,
        OnlineOROfflineMethod.OFFLINE
        """
        return self.__referent_dict_method

    @referent_dict_method.setter
    def referent_dict_method(self, referent_dict_method: str):
        self.__referent_dict_method = referent_dict_method

    @property
    def vo_search_method(self):
        """
        VerbOcean search method type, one of: OnlineOROfflineMethod.ONLINE,
        OnlineOROfflineMethod.OFFLINE
        """
        return self.__vo_search_method

    @vo_search_method.setter
    def vo_search_method(self, vo_search_method: str):
        self.__vo_search_method = vo_search_method
