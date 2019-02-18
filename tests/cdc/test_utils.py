from nlp_architect.common.cdc.mention_data import MentionData


def get_embedd_mentions():
    mentions_json = [
        {
            "coref_chain": "HUM16236184328979740",
            "doc_id": "0",
            "mention_context": [
                "Perennial",
                "party",
                "girl",
                "Tara",
                "Reid",
                "checked",
                "herself",
                "into",
                "Promises",
                "Treatment",
                "Center",
                ",",
                "her",
                "rep",
                "told",
                "People",
                "."
            ],
            "mention_head": "Reid",
            "mention_head_lemma": "reid",
            "mention_head_pos": "PROPN",
            "mention_id": "0",
            "mention_index": -1,
            "mention_ner": "PERSON",
            "mention_type": "HUM",
            "predicted_coref_chain": None,
            "score": -1.0,
            "sent_id": 0,
            "tokens_number": [
                3,
                4
            ],
            "tokens_str": "Tara Reid",
            "topic_id": "1ecb"
        },
        {
            "coref_chain": "HUM16236184328979740",
            "doc_id": "1_12ecb.xml",
            "mention_context": [
                "Tara",
                "Reid",
                "has",
                "checked",
                "into",
                "Promises",
                "Treatment",
                "Center",
                ",",
                "a",
                "prominent",
                "rehab",
                "clinic",
                "in",
                "Los",
                "Angeles",
                "."
            ],
            "mention_head": "Reid",
            "mention_head_lemma": "reid",
            "mention_head_pos": "PROPN",
            "mention_id": "1",
            "mention_index": -1,
            "mention_ner": "PERSON",
            "mention_type": "HUM",
            "predicted_coref_chain": None,
            "score": -1.0,
            "sent_id": 1,
            "tokens_number": [
                0,
                1
            ],
            "tokens_str": "Tara Reid",
            "topic_id": "1ecb"
        },
        {
            "coref_chain": "Singleton_LOC_8_1_12ecb",
            "doc_id": "1_12ecb.xml",
            "mention_context": [
                "Tara",
                "Reid",
                "has",
                "checked",
                "into",
                "Promises",
                "Treatment",
                "Center",
                ",",
                "a",
                "prominent",
                "rehab",
                "clinic",
                "in",
                "Los",
                "Angeles",
                "."
            ],
            "mention_head": "in",
            "mention_head_lemma": "in",
            "mention_head_pos": "ADP",
            "mention_id": "2",
            "mention_index": -1,
            "mention_ner": None,
            "mention_type": "LOC",
            "predicted_coref_chain": None,
            "score": -1.0,
            "sent_id": 1,
            "tokens_number": [
                13,
                14,
                15
            ],
            "tokens_str": "in Los Angeles",
            "topic_id": "1ecb"
        },
        {
            "coref_chain": "HUM16236184328979740",
            "doc_id": "0",
            "mention_context": None,
            "mention_head": "Reid",
            "mention_head_lemma": "reid",
            "mention_head_pos": "PROPN",
            "mention_id": "3",
            "mention_ner": "PERSON",
            "mention_type": "HUM",
            "predicted_coref_chain": None,
            "score": -1.0,
            "sent_id": 0,
            "tokens_number": [
                3,
                4
            ],
            "tokens_str": "Tara Reid",
            "topic_id": "1ecb"
        }
    ]

    mentions = list()
    for json in mentions_json:
        mentions.append(MentionData.read_json_mention_data_line(json))

    return mentions


def get_wiki_mentions():
    mentions_json = [
        {
            "mention_id": "0",
            "tokens_str": "Ellen DeGeneres",
            "topic_id": "1ecb"
        },
        {
            "mention_id": "1",
            "tokens_str": "television host",
            "topic_id": "1ecb"
        },
        {
            "mention_id": "2",
            "tokens_str": "Los Angeles",
            "topic_id": "1ecb"
        }
    ]

    mentions = list()
    for json in mentions_json:
        mentions.append(MentionData.read_json_mention_data_line(json))

    return mentions


def get_compute_mentions():
    mentions_json = [
        {
            "mention_id": "0",
            "tokens_str": "Exact String",
            "topic_id": "1ecb"
        },
        {
            "mention_id": "1",
            "tokens_str": "Exact Same Head String",
            "topic_id": "1ecb"
        },
        {
            "mention_id": "2",
            "tokens_str": "Nothing",
            "topic_id": "1ecb"
        }
    ]

    mentions = list()
    for json in mentions_json:
        mentions.append(MentionData.read_json_mention_data_line(json))

    return mentions


def get_wordnet_mentions():
    mentions_json = [
        {
            "mention_id": "0",
            "tokens_str": "play",
            "topic_id": "1ecb"
        },
        {
            "mention_id": "1",
            "tokens_str": "game",
            "topic_id": "1ecb"
        },
        {
            "mention_id": "2",
            "tokens_str": "Chair",
            "topic_id": "1ecb"
        }
    ]

    mentions = list()
    for json in mentions_json:
        mentions.append(MentionData.read_json_mention_data_line(json))

    return mentions
