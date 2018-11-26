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

import logging

from nlp_architect.data.cdc_resources.relations.computed_relation_extraction import \
    ComputedRelationExtraction
from nlp_architect.data.cdc_resources.relations.referent_dict_relation_extraction import \
    ReferentDictRelationExtraction
from nlp_architect.data.cdc_resources.relations.relation_types_enums import RelationType
from nlp_architect.data.cdc_resources.relations.verbocean_relation_extraction import \
    VerboceanRelationExtraction
from nlp_architect.data.cdc_resources.relations.wikipedia_relation_extraction import \
    WikipediaRelationExtraction
from nlp_architect.data.cdc_resources.relations.within_doc_coref_extraction import WithinDocCoref
from nlp_architect.data.cdc_resources.relations.word_embedding_relation_extraction import \
    WordEmbeddingRelationExtraction
from nlp_architect.data.cdc_resources.relations.wordnet_relation_extraction import \
    WordnetRelationExtraction

logger = logging.getLogger(__name__)


class CDCSettings(object):
    def __init__(self, resources, event_coref_config, entity_coref_config):
        self.wiki = None
        self.vo = None
        self.embeds = None
        self.ref_dict = None
        self.context2vec_model = None
        self.wordnet = None
        self.within_doc = None
        self.event_config = event_coref_config
        self.entity_config = entity_coref_config
        self.cdc_resources = resources

        self.load_modules()

    def load_modules(self):
        relations = set()
        for sieve in self.event_config.sieves_order:
            relations.add(sieve[1])
        for sieve in self.entity_config.sieves_order:
            relations.add(sieve[1])

        if any('WIKIPEDIA' in relation.name for relation in relations):
            self.wiki = WikipediaRelationExtraction(self.cdc_resources.wiki_search_method,
                                                    wiki_file=self.cdc_resources.wiki_folder,
                                                    host=self.cdc_resources.elastic_host,
                                                    port=self.cdc_resources.elastic_port,
                                                    index=self.cdc_resources.elastic_index)
        if RelationType.WORD_EMBEDDING_MATCH in relations:
            self.embeds = WordEmbeddingRelationExtraction(self.cdc_resources.embed_search_method,
                                                          glove_file=self.cdc_resources.glove_file,
                                                          elmo_file=self.cdc_resources.elmo_file)
        if RelationType.VERBOCEAN_MATCH in relations:
            self.vo = VerboceanRelationExtraction(self.cdc_resources.vo_search_method,
                                                  self.cdc_resources.vo_dict_file)
        if RelationType.REFERENT_DICT in relations:
            self.ref_dict = ReferentDictRelationExtraction(self.cdc_resources.referent_dict_method,
                                                           self.cdc_resources.referent_dict_file)
        if RelationType.WITHIN_DOC_COREF in relations:
            self.within_doc = WithinDocCoref(self.cdc_resources.wd_file)
        if any('WORDNET' in relation.name for relation in relations):
            self.wordnet = WordnetRelationExtraction(self.cdc_resources.wn_search_method,
                                                     self.cdc_resources.wn_folder)

    def get_module_from_relation(self, relation_type):
        if RelationType.WITHIN_DOC_COREF == relation_type:
            ret_model = self.within_doc
        elif relation_type in [
                RelationType.EXACT_STRING, RelationType.SAME_HEAD_LEMMA,
                RelationType.SAME_HEAD_LEMMA_RELAX,
                RelationType.FUZZY_HEAD_FIT,
                RelationType.FUZZY_FIT]:
            ret_model = ComputedRelationExtraction()
        elif relation_type in [
                RelationType.WORDNET_SAME_SYNSET_ENTITY,
                RelationType.WORDNET_PARTIAL_SYNSET_MATCH,
                RelationType.WORDNET_DERIVATIONALLY,
                RelationType.WORDNET_SAME_SYNSET_EVENT]:
            ret_model = self.wordnet
        elif RelationType.WORD_EMBEDDING_MATCH == relation_type:
            ret_model = self.embeds
        elif relation_type in [
                RelationType.WIKIPEDIA_REDIRECT_LINK,
                RelationType.WIKIPEDIA_DISAMBIGUATION,
                RelationType.WIKIPEDIA_BE_COMP,
                RelationType.WIKIPEDIA_CATEGORY,
                RelationType.WIKIPEDIA_TITLE_PARENTHESIS]:
            ret_model = self.wiki
        elif RelationType.REFERENT_DICT == relation_type:
            ret_model = self.ref_dict
        elif RelationType.VERBOCEAN_MATCH == relation_type:
            ret_model = self.vo
        else:
            raise Exception('Not a Supported RelationType-' + relation_type)

        return ret_model
