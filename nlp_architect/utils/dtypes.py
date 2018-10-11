# -*- coding: utf-8 -*-

"""
Created on 2018/10/7 上午12:07

@author: xujiang@baixing.com

"""

import six
import numpy as np

import tensorflow as tf

__all__ = [
    "get_tf_dtype",
    "is_callable",
    "is_str",
    "is_placeholder",
    "maybe_hparams_to_dict",
    "compat_as_text"
]

def get_tf_dtype(dtype): # pylint: disable=too-many-return-statements
    """Returns equivalent tf dtype.
    Args:
        dtype: A str, python numeric or string type, numpy data type, or
            tf dtype.
    Returns:
        The corresponding tf dtype.
    """
    if dtype in {'float', 'float32', 'tf.float32', float,
                 np.float32, tf.float32}:
        return tf.float32
    elif dtype in {'float64', 'tf.float64', np.float64, np.float_, tf.float64}:
        return tf.float64
    elif dtype in {'float16', 'tf.float16', np.float16, tf.float16}:
        return tf.float16
    elif dtype in {'int', 'int32', 'tf.int32', int, np.int32, tf.int32}:
        return tf.int32
    elif dtype in {'int64', 'tf.int64', np.int64, tf.int64}:
        return tf.int64
    elif dtype in {'int16', 'tf.int16', np.int16, tf.int16}:
        return tf.int16
    elif dtype in {'bool', 'tf.bool', bool, np.bool_, tf.bool}:
        return tf.bool
    elif dtype in {'string', 'str', 'tf.string', str, np.str, tf.string}:
        return tf.string
    # try:
    #     if dtype == {'unicode', }:
    #         return tf.string
    # except NameError:
    #     pass

    raise ValueError(
        "Unsupported conversion from type {} to tf dtype".format(str(dtype)))

def is_callable(x):
    """Return `True` if :attr:`x` is callable.
    """
    try:
        _is_callable = callable(x)
    except: # pylint: disable=bare-except
        _is_callable = hasattr(x, '__call__')
    return _is_callable

def is_str(x):
    """Returns `True` if :attr:`x` is either a str or unicode. Returns `False`
    otherwise.
    """
    return isinstance(x, six.string_types)

def is_placeholder(x):
    """Returns `True` if :attr:`x` is a :tf_main:`tf.placeholder <placeholder>`
    or :tf_main:`tf.placeholder_with_default <placeholder_with_default>`.
    """
    try:
        return x._ops.type in ['Placeholder', 'PlaceholderWithDefault']
    except: # pylint: disable=bare-except
        return False

def maybe_hparams_to_dict(hparams):
    """If :attr:`hparams` is an instance of :class:`~texar.HParams`,
    converts it to a `dict` and returns. If :attr:`hparams` is a `dict`,
    returns as is.
    """
    if hparams is None:
        return None
    if isinstance(hparams, dict):
        return hparams
    return hparams.todict()

def _maybe_list_to_array(str_list, dtype_as):
    if isinstance(dtype_as, (list, tuple)):
        return type(dtype_as)(str_list)
    elif isinstance(dtype_as, np.ndarray):
        return np.array(str_list)
    else:
        return str_list

def compat_as_text(str_):
    """Converts strings into `unicode` (Python 2) or `str` (Python 3).
    Args:
        str\_: A string or other data types convertible to string, or an
            `n`-D numpy array or (possibly nested) list of such elements.
    Returns:
        The converted strings of the same structure/shape as :attr:`str_`.
    """
    def _recur_convert(s):
        if isinstance(s, (list, tuple, np.ndarray)):
            s_ = [_recur_convert(si) for si in s]
            return _maybe_list_to_array(s_, s)
        else:
            try:
                return tf.compat.as_text(s)
            except TypeError:
                return tf.compat.as_text(str(s))

    text = _recur_convert(str_)

    return text