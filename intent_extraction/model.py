from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from keras import Input, Model
from keras.layers import (LSTM, Bidirectional, Dense, Dropout, Embedding,
                          RepeatVector, TimeDistributed, concatenate)
from keras.models import load_model


class IntentExtractionModel(object):
    def __init__(self):
        self.model = None

    def fit(self, x, y, epochs=1, batch_size=1, callbacks=None, validation=None):
        assert self.model, 'Model was not initialized'
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, shuffle=True,
                       validation_data=validation,
                       callbacks=callbacks)

    def predict(self, x, batch_size=1):
        assert self.model, 'Model was not initialized'
        return self.model.predict(x, batch_size=batch_size)

    def save(self, path):
        assert self.model, 'Model was not initialized'
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

    @property
    def input_shape(self):
        return self.model.layers[0].input_shape

    @staticmethod
    def create_input_embed(sentence_len, is_extern_emb, token_emb_size, vocab_size):
        if is_extern_emb:
            in_layer = e_layer = Input(shape=(sentence_len, token_emb_size,),
                                       dtype='float32', name='tokens_input')
        else:
            in_layer = Input(shape=(sentence_len,),
                             dtype='int32', name='tokens_input')
            e_layer = Embedding(vocab_size, token_emb_size,
                                input_length=sentence_len,
                                name='embedding_layer')(in_layer)
        return in_layer, e_layer


class JointSequentialLSTM(IntentExtractionModel):
    def __init__(self):
        super(JointSequentialLSTM, self).__init__()

    def build(self,
              sentence_length,
              vocab_size,
              tag_labels,
              intent_labels,
              token_emb_size=100,
              tagger_hidden=100,
              tagger_dropout=0.5,
              intent_classifier_hidden=100,
              emb_model_path=None):
        tokens_input, token_emb = self.create_input_embed(sentence_length,
                                                          emb_model_path is not None,
                                                          token_emb_size,
                                                          vocab_size)
        intent_enc = Bidirectional(LSTM(intent_classifier_hidden))(token_emb)
        intent_out = Dense(intent_labels, activation='softmax',
                           name='intent_classifier')(intent_enc)
        intent_vec_rep = RepeatVector(sentence_length)(intent_out)

        slot_emb = Bidirectional(LSTM(tagger_hidden, return_sequences=True))(token_emb)
        tagger_features = concatenate([slot_emb, intent_vec_rep], axis=-1)
        tagger = Bidirectional(
            LSTM(tagger_hidden, return_sequences=True))(tagger_features)
        tagger = Dropout(tagger_dropout)(tagger)
        tagger_out = TimeDistributed(
            Dense(tag_labels, activation='softmax'),
            name='slot_tag_classifier')(tagger)

        self.model = Model(inputs=tokens_input, outputs=[
            intent_out, tagger_out])
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                           loss_weights=[1., 1.], metrics=['categorical_accuracy'])


class EncDecTaggerModel(IntentExtractionModel):
    def __init__(self):
        super(EncDecTaggerModel, self).__init__()

    def build(self,
              sentence_length,
              vocab_size,
              tag_labels,
              token_emb_size=100,
              encoder_depth=1,
              decoder_depth=1,
              lstm_hidden_size=100,
              encoder_dropout=0.5,
              decoder_dropout=0.5,
              emb_model_path=None):
        tokens_input, token_emb = self.create_input_embed(sentence_length,
                                                          emb_model_path is not None,
                                                          token_emb_size,
                                                          vocab_size)
        benc_in = token_emb
        assert encoder_depth > 0, 'Encoder depth must be > 0'
        for i in range(encoder_depth):
            bencoder = LSTM(lstm_hidden_size, return_sequences=True, return_state=True,
                            go_backwards=True, dropout=encoder_dropout,
                            name='encoder_blstm_{}'.format(i))(benc_in)
            benc_in = bencoder[0]
        b_states = bencoder[1:]
        benc_h, bene_c = b_states

        decoder_inputs = token_emb
        assert decoder_depth > 0, 'Decoder depth must be > 0'
        for i in range(decoder_depth):
            decoder = LSTM(lstm_hidden_size, return_sequences=True,
                           name='decoder_lstm_{}'.format(i))(decoder_inputs,
                                                             initial_state=[benc_h,
                                                                            bene_c])
            decoder_inputs = decoder
        decoder_outputs = Dropout(decoder_dropout)(decoder)
        decoder_predictions = TimeDistributed(
            Dense(tag_labels, activation='softmax'),
            name='decoder_classifier')(decoder_outputs)

        self.model = Model(tokens_input, decoder_predictions)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
