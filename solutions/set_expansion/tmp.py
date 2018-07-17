import spacy

nlp = spacy.load('en_core_web_sm', disable=['textcat', 'ner',])
doc = nlp(u'The west Belfast area  also saw the formation  in 1973 of '
          u'the Ulster Freedom Fighters (UFF) by former Harding Smith ally John White.')

spans = list()
for span in doc.noun_chunks:
    print(span)