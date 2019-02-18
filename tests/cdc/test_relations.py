from nlp_architect.data.cdc_resources.embedding.embed_elmo import ElmoEmbedding
from nlp_architect.data.cdc_resources.relations.computed_relation_extraction import \
    ComputedRelationExtraction
from nlp_architect.data.cdc_resources.relations.relation_types_enums import EmbeddingMethod, \
    RelationType
from nlp_architect.data.cdc_resources.relations.wikipedia_relation_extraction import \
    WikipediaRelationExtraction
from nlp_architect.data.cdc_resources.relations.word_embedding_relation_extraction import \
    WordEmbeddingRelationExtraction
from nlp_architect.data.cdc_resources.relations.wordnet_relation_extraction import \
    WordnetRelationExtraction
from tests.cdc.test_utils import get_embedd_mentions, get_wiki_mentions, get_compute_mentions, \
    get_wordnet_mentions


def test_wiki_online():
    mentions = get_wiki_mentions()
    wiki = WikipediaRelationExtraction()

    assert not wiki.extract_all_relations(
        mentions[0], mentions[0]).isdisjoint(set([RelationType.WIKIPEDIA_CATEGORY,
                                                  RelationType.WIKIPEDIA_REDIRECT_LINK,
                                                  RelationType.WIKIPEDIA_BE_COMP]))

    assert not wiki.extract_all_relations(
        mentions[0], mentions[1]).isdisjoint(set([RelationType.WIKIPEDIA_CATEGORY,
                                                  RelationType.WIKIPEDIA_REDIRECT_LINK,
                                                  RelationType.WIKIPEDIA_BE_COMP]))

    assert wiki.extract_all_relations(
        mentions[0], mentions[2]).pop() == RelationType.NO_RELATION_FOUND


def test_compute_relations():
    mentions = get_compute_mentions()
    compute = ComputedRelationExtraction()

    assert not compute.extract_all_relations(
        mentions[0], mentions[0]).isdisjoint(set([RelationType.EXACT_STRING]))

    assert not compute.extract_all_relations(
        mentions[0], mentions[1]).isdisjoint(set([RelationType.SAME_HEAD_LEMMA,
                                                  RelationType.FUZZY_HEAD_FIT]))

    assert compute.extract_all_relations(
        mentions[0], mentions[2]).pop() == RelationType.NO_RELATION_FOUND


def test_wordnet_relations():
    mentions = get_wordnet_mentions()
    wordnet = WordnetRelationExtraction()

    assert not wordnet.extract_all_relations(
        mentions[0], mentions[1]).isdisjoint(set([RelationType.WORDNET_DERIVATIONALLY]))

    assert wordnet.extract_all_relations(
        mentions[0], mentions[2]).pop() == RelationType.NO_RELATION_FOUND


def test_elmo_offline():
    mentions = get_embedd_mentions()
    embeder = ElmoEmbedding()
    for mention in mentions:
        embeder.get_head_feature_vector(mention)

    elmo_embeddings = WordEmbeddingRelationExtraction(EmbeddingMethod.ELMO_OFFLINE, elmo_file=None)
    elmo_embeddings.embedding.embeder = embeder.cache

    assert elmo_embeddings.extract_all_relations(
        mentions[0], mentions[0]).pop() == RelationType.WORD_EMBEDDING_MATCH

    assert elmo_embeddings.extract_all_relations(
        mentions[0], mentions[1]).pop() == RelationType.WORD_EMBEDDING_MATCH

    assert elmo_embeddings.extract_all_relations(
        mentions[0], mentions[2]).pop() == RelationType.NO_RELATION_FOUND

    assert elmo_embeddings.extract_all_relations(
        mentions[0], mentions[3]).pop() == RelationType.WORD_EMBEDDING_MATCH


def test_elmo_online():
    mentions = get_embedd_mentions()
    elmo_embeddings = WordEmbeddingRelationExtraction(EmbeddingMethod.ELMO)

    assert elmo_embeddings.extract_all_relations(
        mentions[0], mentions[0]).pop() == RelationType.WORD_EMBEDDING_MATCH

    assert elmo_embeddings.extract_all_relations(
        mentions[0], mentions[1]).pop() == RelationType.WORD_EMBEDDING_MATCH

    assert elmo_embeddings.extract_all_relations(
        mentions[0], mentions[2]).pop() == RelationType.NO_RELATION_FOUND

    assert elmo_embeddings.extract_all_relations(
        mentions[0], mentions[3]).pop() == RelationType.WORD_EMBEDDING_MATCH
