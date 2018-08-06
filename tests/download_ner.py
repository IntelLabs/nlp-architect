from nlp_architect.api.ner_api import NerApi


def download():
    NerApi(prompt=False)  # to download NER model without prompt
