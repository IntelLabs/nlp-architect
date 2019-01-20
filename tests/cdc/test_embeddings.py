from nlp_architect.data.cdc_resources.embedding.embed_elmo import ElmoEmbedding
from nlp_architect.data.cdc_resources.relations.relation_types_enums import EmbeddingMethod, \
    RelationType
from nlp_architect.data.cdc_resources.relations.word_embedding_relation_extraction import \
    WordEmbeddingRelationExtraction
from tests.cdc.test_utils import get_mentions


def test_elmo_offline():
    mentions = get_mentions()
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

    print()


def test_elmo_online():
    mentions = get_mentions()
    elmo_embeddings = WordEmbeddingRelationExtraction(EmbeddingMethod.ELMO)

    assert elmo_embeddings.extract_all_relations(
        mentions[0], mentions[0]).pop() == RelationType.WORD_EMBEDDING_MATCH

    assert elmo_embeddings.extract_all_relations(
        mentions[0], mentions[1]).pop() == RelationType.WORD_EMBEDDING_MATCH

    assert elmo_embeddings.extract_all_relations(
        mentions[0], mentions[2]).pop() == RelationType.NO_RELATION_FOUND

    assert elmo_embeddings.extract_all_relations(
        mentions[0], mentions[3]).pop() == RelationType.WORD_EMBEDDING_MATCH
