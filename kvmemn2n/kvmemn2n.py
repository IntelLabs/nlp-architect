from __future__ import division
from __future__ import print_function
import ngraph as ng
from ngraph.frontends.neon import Layer, LookupTable
from ngraph.frontends.neon.axis import shadow_axes_map
from ngraph.frontends.neon import Layer
from ngraph.frontends.neon import GaussianInit
from ngraph.frontends.neon.graph import SubGraph
from util import position_encoding

# Labels should be added as metadata on specific ops and variables
# Hopefully these can be used to efficiently display and filter the
# computational graph
LABELS = {"weight": "weight",
          "bias": "bias"}


class ModifiedLookupTable(Layer):
    """
    Lookup table layer that often is used as word embedding layer.

    Modified from the default LookupTable implementation to support multiple axis lookups.

    Args:
        vocab_size (int): the vocabulary size
        embed_dim (int): the size of embedding vector
        init (Initializor): initialization function
        update (bool): if the word vectors get updated through training
        pad_idx (int): by knowing the pad value, the update will make sure always
                       have the vector representing pad value to be 0s.
    """

    def __init__(self, vocab_size, embed_dim, init, update=True, pad_idx=None,
                 **kwargs):
        super(ModifiedLookupTable, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.init = init
        self.update = update
        self.pad_idx = pad_idx
        self.W = None

    def lut_init(self, axes, pad_word_axis, pad_idx):
        """
        Initialization function for the lut.
        After using the initialization to fill the whole array, set the part that represents
        padding to be 0.
        """
        init_w = self.init(axes)
        if axes.index(pad_word_axis) is 0:
            init_w[pad_idx] = 0
        else:
            init_w[:, pad_idx] = 0
        return init_w

    @SubGraph.scope_op_creation
    def __call__(self, in_obj, **kwargs):
        """
        Arguments:
            in_obj (Tensor): object that provides the lookup indices
        """
        in_obj = ng.flatten(in_obj)
        in_axes = in_obj.axes

        # label lut_v_axis as shadow axis for initializers ... once #1158 is
        # in, shadow axis will do more than just determine fan in/out for
        # initializers.
        self.lut_v_axis = ng.make_axis(self.vocab_size).named('V')
        self.axes_map = shadow_axes_map([self.lut_v_axis])
        self.lut_v_axis = list(self.axes_map.values())[0]

        self.lut_f_axis = ng.make_axis(self.embed_dim).named('F')

        self.w_axes = ng.make_axes([self.lut_v_axis, self.lut_f_axis])
        self.lut_o_axes = in_axes | ng.make_axes([self.lut_f_axis])
        self.o_axes = ng.make_axes([self.lut_f_axis]) | in_axes[0].axes

        if not self.initialized:
            self.W = ng.variable(
                axes=self.w_axes,
                initial_value=self.lut_init(
                    self.w_axes,
                    self.lut_v_axis,
                    self.pad_idx),
                metadata={
                    "label": LABELS["weight"]},
            ).named('LutW')

        lut_result = ng.lookuptable(
            self.W,
            in_obj,
            self.lut_o_axes,
            update=self.update,
            pad_idx=self.pad_idx)
        return ng.map_roles(ng.unflatten(lut_result), self.axes_map)


class KVMemN2N(Layer):
    def __init__(self, num_iterations, batch_size, emb_size, nhops,
                 story_length, memory_size, vocab_size, vocab_axis, use_v_luts):

        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.nhops = nhops
        self.story_length = story_length
        self.memory_size = memory_size
        self.vocab_size = vocab_size
        self.use_v_luts = use_v_luts

        # Create graph
        # Make axes
        self.batch_axis = ng.make_axis(length=batch_size, name='N')
        self.sentence_axis = ng.make_axis(length=story_length, name='sentence_axis')
        self.sentence_rec_axis = ng.make_axis(length=story_length, name='REC')
        self.memory_axis = ng.make_axis(length=memory_size, name='memory_axis')

        self.val_len_axis = ng.make_axis(length=1, name='REC')

        self.embedding_axis = ng.make_axis(length=emb_size, name='F')

        self.vocab_axis = vocab_axis

        # weight initializationn
        self.init = GaussianInit(mean=0.0, std=0.1)
        # Create constant position encoding tensor to multiply elementwise with embedded words
        self.pos_enc = position_encoding(self.sentence_rec_axis, self.embedding_axis)

        # Weight sharing
        self.LUT_A = ModifiedLookupTable(self.vocab_size, self.emb_size, self.init, update=True,
                                         pad_idx=0, name='LUT_A')
        if use_v_luts:
            self.LUTs_C = [ModifiedLookupTable(self.vocab_size, self.emb_size, self.init,
                           update=True, pad_idx=0) for n in range(self.nhops)]

    def __call__(self, inputs):
        query = ng.cast_axes(inputs['query'], [self.batch_axis, self.sentence_rec_axis])

        # Query embedding [batch, sentence_axis, F]
        q_emb = self.LUT_A(query)

        # Multiply by position encoding and sum
        u_0 = ng.sum(q_emb * self.pos_enc, reduction_axes=[self.sentence_rec_axis])  # [batch, F]

        # Start a list of the internal states of the model.
        # Will be appended to after each memory hop
        u = [u_0]

        for hopn in range(self.nhops):
            keys = ng.cast_axes(inputs['keys'], [self.batch_axis, self.memory_axis,
                                self.sentence_rec_axis])
            value = ng.cast_axes(inputs['values'], [self.batch_axis, self.memory_axis,
                                 self.val_len_axis])

            # Embed keys
            m_emb_A = self.LUT_A(keys)
            m_A = ng.sum(m_emb_A * self.pos_enc,
                         reduction_axes=[self.sentence_rec_axis])  # [batch, memory_axis, F]

            # Compute scalar similarity between internal state and each memory
            # Equivalent to dot product between u[-1] and each memory in m_A
            dotted = ng.sum(u[-1] * m_A, reduction_axes=[self.embedding_axis])

            probs = ng.softmax(dotted, self.memory_axis)  # [batch, memory_axis]

            # Embed values with same embedding as keys, or new LUTs
            if self.use_v_luts:
                m_emb_C = self.LUTs_C[hopn](value)
            else:
                m_emb_C = self.LUT_A(value)

            m_C = ng.sum(m_emb_C * self.pos_enc, reduction_axes=[self.sentence_rec_axis])

            # Compute weighted sum of output embeddings
            o_k = ng.sum(probs * m_C, reduction_axes=[self.memory_axis])  # [batch, F]

            u_k = u[-1] + o_k  # [batch, F]

            # Add new internal state
            u.append(u_k)

        # Compute predicted answer from product of final internal state and final LUT weight matrix
        if self.use_v_luts:
            a_logits = ng.dot(self.LUTs_C[-1].W, u[-1])  # [batch, V]
        else:
            a_logits = ng.dot(self.LUT_A.W, u[-1])  # [batch, V]
        # rename V to vocab_axis to match answer
        a_logits = ng.cast_axes(a_logits, [self.vocab_axis, self.batch_axis])
        a_pred = ng.softmax(a_logits, self.vocab_axis)

        return a_pred, a_logits
