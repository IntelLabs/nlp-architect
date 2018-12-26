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

from nlp_architect import LIBRARY_ROOT
from nlp_architect.common.cdc.mention_data import MentionDataLight
from nlp_architect.data.cdc_resources.relations.computed_relation_extraction import \
    ComputedRelationExtraction
from nlp_architect.data.cdc_resources.relations.referent_dict_relation_extraction import \
    ReferentDictRelationExtraction
from nlp_architect.data.cdc_resources.relations.relation_types_enums import \
    RelationType, EmbeddingMethod
from nlp_architect.data.cdc_resources.relations.verbocean_relation_extraction import \
    VerboceanRelationExtraction
from nlp_architect.data.cdc_resources.relations.wikipedia_relation_extraction import \
    WikipediaRelationExtraction
from nlp_architect.data.cdc_resources.relations.word_embedding_relation_extraction import \
    WordEmbeddingRelationExtraction
from nlp_architect.data.cdc_resources.relations.wordnet_relation_extraction import \
    WordnetRelationExtraction


def run_example():
    logger.info('Running relation extraction example......')
    computed = ComputedRelationExtraction()
    ref_dict = ReferentDictRelationExtraction(ref_dict=LIBRARY_ROOT + '/datasets/coref.dict1.tsv')
    vo = VerboceanRelationExtraction(
        vo_file=LIBRARY_ROOT + '/datasets/verbocean.unrefined.2004-05-20.txt')
    wiki = WikipediaRelationExtraction()
    wn = WordnetRelationExtraction()
    embed = WordEmbeddingRelationExtraction(method=EmbeddingMethod.ELMO)

    mention_x1 = MentionDataLight(
        'IBM',
        mention_context='IBM manufactures and markets computer hardware, middleware and software')
    mention_y1 = MentionDataLight(
        'International Business Machines',
        mention_context='International Business Machines Corporation is an '
                        'American multinational information technology company')

    computed_relations = computed.extract_all_relations(mention_x1, mention_y1)
    ref_dict_relations = ref_dict.extract_all_relations(mention_x1, mention_y1)
    vo_relations = vo.extract_all_relations(mention_x1, mention_y1)
    wiki_relations = wiki.extract_all_relations(mention_x1, mention_y1)
    embed_relations = embed.extract_all_relations(mention_x1, mention_y1)
    wn_relaions = wn.extract_all_relations(mention_x1, mention_y1)

    if RelationType.NO_RELATION_FOUND in computed_relations:
        logger.info('No Computed relation found')
    else:
        logger.info('Found Computed relations-%s', str(list(computed_relations)))

    if RelationType.NO_RELATION_FOUND in ref_dict_relations:
        logger.info('No Referent-Dict relation found')
    else:
        logger.info('Found Referent-Dict relations-%s', str(list(ref_dict_relations)))

    if RelationType.NO_RELATION_FOUND in vo_relations:
        logger.info('No Verb-Ocean relation found')
    else:
        logger.info('Found Verb-Ocean relations-%s', str(list(vo_relations)))

    if RelationType.NO_RELATION_FOUND in wiki_relations:
        logger.info('No Wikipedia relation found')
    else:
        logger.info('Found Wikipedia relations-%s', str(wiki_relations))
    if RelationType.NO_RELATION_FOUND in embed_relations:
        logger.info('No Embedded relation found')
    else:
        logger.info('Found Embedded relations-%s', str(list(embed_relations)))
    if RelationType.NO_RELATION_FOUND in wn_relaions:
        logger.info('No Wordnet relation found')
    else:
        logger.info('Found Wordnet relations-%s', str(wn_relaions))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    run_example()
