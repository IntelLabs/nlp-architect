import numpy as np
import ngraph as ng


def position_encoding(sentence_axis, embedding_axis):
    """
    Position Encoding described in section 4.1 [1]
    """
    sentence_size = sentence_axis.length
    embedding_size = embedding_axis.length
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_size + 1) / 2) * (j - (sentence_size + 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    encoding = np.transpose(encoding)

    return ng.constant(encoding, axes=[sentence_axis, embedding_axis])
