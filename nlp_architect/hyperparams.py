# -*- coding: utf-8 -*-

"""
Created on 2018/10/7 上午12:05

@author: xujiang@baixing.com

"""

import copy
import json

from nlp_architect.utils.dtypes import is_callable

__all__ = [
    "HParams"
]

def _type_name(value):
    return type(value).__name__

class HParams(object):
    """A class that maintains hyperparameters for configing Texar modules.
    The class has several useful features:
    - **Auto-completion of missing values.** Users can specify only a subset of\
    hyperparameters they care about. Other hyperparameters will automatically\
    take the default values. The auto-completion performs **recursively** so \
    that hyperparameters taking `dict` values will also be auto-completed \
    **All Texar modules** provide a \
    :meth:`default_hparams` containing allowed hyperparameters and their \
    default values. For example
        .. code-block:: python
            ## Recursive auto-completion
            default_hparams = {"a": 1, "b": {"c": 2, "d": 3}}
            hparams = {"b": {"c": 22}}
            hparams_ = HParams(hparams, default_hparams)
            hparams_.todict() == {"a": 1, "b": {"c": 22, "d": 3}}
                # "a" and "d" are auto-completed
            ## All Texar modules have built-in `default_hparams`
            hparams = {"dropout_rate": 0.1}
            emb = tx.modules.WordEmbedder(hparams=hparams, ...)
            emb.hparams.todict() == {
                "dropout_rate": 0.1,  # provided value
                "dim": 100            # default value
                ...
            }
    - **Automatic typecheck.** For most hyperparameters, provided value must\
    have the same or compatible dtype with the default value. HParams does\
    necessary typecheck, and raises Error if improper dtype is provided.\
    Also, hyperparameters not listed in `default_hparams` are not allowed,\
    except for "kwargs" as detailed below.
    - **Flexible dtype for specified hyperparameters.**  Some hyperparameters\
    may allow different dtypes of values.
        - Hyperparameters named "type" are not typechecked.\
        For example, in :func:`~texar.core.get_rnn_cell`, hyperparameter \
        `"type"` can take value of an RNNCell class, its string name of module \
        path, or an RNNCell class instance. (String name or module path is \
        allowd so that users can specify the value in YAML config files.)
        - For other hyperparameters, list them\
        in the "@no_typecheck" field in `default_hparams` to skip typecheck. \
        For example, in :func:`~texar.core.get_rnn_cell`, hyperparameter \
        "*_keep_prob" can be set to either a `float` or a `tf.placeholder`.
    - **Special flexibility of keyword argument hyparameters.** \
    Hyperparameters named "kwargs" are used as keyword arguments for a class\
    constructor or a function call. Such hyperparameters take a `dict`, and \
    users can add arbitrary valid keyword arguments to the dict. For example:
        .. code-block:: python
            default_rnn_cell_hparams = {
                "type": "LSTMCell",
                "kwargs": {"num_units": 256}
                # Other hyperparameters
                ...
            }
            my_hparams = {
                "kwargs" {
                    "num_units": 123,
                    "forget_bias": 0.0         # Other valid keyword arguments
                    "activation": "tf.nn.relu" # for LSTMCell constructor
                }
            }
            _ = HParams(my_hparams, default_rnn_cell_hparams)
    - **Rich interfaces.** An HParams instance provides rich interfaces for\
    accessing, updating, or adding hyperparameters.
        .. code-block:: python
            hparams = HParams(my_hparams, default_hparams)
            # Access
            hparams.type == hparams["type"]
            # Update
            hparams.type = "GRUCell"
            hparams.kwargs = { "num_units": 100 }
            hparams.kwargs.num_units == 100
            # Add new
            hparams.add_hparam("index", 1)
            hparams.index == 1
            # Convert to `dict` (recursively)
            type(hparams.todic()) == dict
            # I/O
            pickle.dump(hparams, "hparams.dump")
            with open("hparams.dump", 'rb') as f:
                hparams_loaded = pickle.load(f)
    Args:
        hparams: A `dict` or an `HParams` instance containing hyperparameters.
            If `None`, all hyperparameters are set to default values.
        default_hparams (dict): Hyperparameters with default values. If `None`,
            Hyperparameters are fully defined by :attr:`hparams`.
        allow_new_hparam (bool): If `False` (default), :attr:`hparams` cannot
            contain hyperparameters that are not included in
            :attr:`default_hparams`, except for the case of :attr:`"kwargs"` as
            above.
    """
    # - The default hyperparameters in :attr:`"kwargs"` are used (for typecheck\
    # and complementing missing hyperparameters) only when :attr:`"type"` \
    # takes default value (i.e., missing in :attr:`hparams` or set to \
    # the same value with the default). In this case :attr:`kwargs` allows to \
    # contain new keys not included in :attr:`default_hparams["kwargs"]`.
    #
    # - If :attr:`"type"` is set to an other \
    # value and :attr:`"kwargs"` is missing in :attr:`hparams`, \
    # :attr:`"kwargs"` is set to an empty dictionary.

    def __init__(self, hparams, default_hparams, allow_new_hparam=False):
        if isinstance(hparams, HParams):
            hparams = hparams.todict()
        if default_hparams is not None:
            parsed_hparams = self._parse(
                hparams, default_hparams, allow_new_hparam)
        else:
            parsed_hparams = self._parse(hparams, hparams)
        super(HParams, self).__setattr__('_hparams', parsed_hparams)

    @staticmethod
    def _parse(hparams, # pylint: disable=too-many-branches, too-many-statements
               default_hparams,
               allow_new_hparam=False):
        """Parses hyperparameters.
        Args:
            hparams (dict): Hyperparameters. If `None`, all hyperparameters are
                set to default values.
            default_hparams (dict): Hyperparameters with default values.
                If `None`,Hyperparameters are fully defined by :attr:`hparams`.
            allow_new_hparam (bool): If `False` (default), :attr:`hparams`
                cannot contain hyperparameters that are not included in
                :attr:`default_hparams`, except the case of :attr:`"kwargs"`.
        Return:
            A dictionary of parsed hyperparameters. Returns `None` if both
            :attr:`hparams` and :attr:`default_hparams` are `None`.
        Raises:
            ValueError: If :attr:`hparams` is not `None` and
                :attr:`default_hparams` is `None`.
            ValueError: If :attr:`default_hparams` contains "kwargs" not does
                not contains "type".
        """
        if hparams is None and default_hparams is None:
            return None

        if hparams is None:
            return HParams._parse(default_hparams, default_hparams)

        if default_hparams is None:
            raise ValueError("`default_hparams` cannot be `None` if `hparams` "
                             "is not `None`.")
        no_typecheck_names = default_hparams.get("@no_typecheck", [])

        if "kwargs" in default_hparams and "type" not in default_hparams:
            raise ValueError("Ill-defined hyperparameter structure: 'kwargs' "
                             "must accompany with 'type'.")

        parsed_hparams = copy.deepcopy(default_hparams)

        # Parse recursively for params of type dictionary that are missing
        # in `hparams`.
        for name, value in default_hparams.items():
            if name not in hparams and isinstance(value, dict):
                if name == "kwargs" and "type" in hparams and \
                        hparams["type"] != default_hparams["type"]:
                    # Set params named "kwargs" to empty dictionary if "type"
                    # takes value other than default.
                    parsed_hparams[name] = HParams({}, {})
                else:
                    parsed_hparams[name] = HParams(value, value)

        # Parse hparams
        for name, value in hparams.items():
            if name not in default_hparams:
                if allow_new_hparam:
                    parsed_hparams[name] = HParams._parse_value(value, name)
                    continue
                else:
                    raise ValueError(
                        "Unknown hyperparameter: %s. Only hyperparameters "
                        "named 'kwargs' hyperparameters can contain new "
                        "entries undefined in default hyperparameters." % name)

            if value is None:
                parsed_hparams[name] = \
                    HParams._parse_value(parsed_hparams[name])

            default_value = default_hparams[name]
            if default_value is None:
                parsed_hparams[name] = HParams._parse_value(value)
                continue

            # Parse recursively for params of type dictionary.
            if isinstance(value, dict):
                if name not in no_typecheck_names \
                        and not isinstance(default_value, dict):
                    raise ValueError(
                        "Hyperparameter '%s' must have type %s, got %s" %
                        (name, _type_name(default_value), _type_name(value)))
                if name == "kwargs":
                    if "type" in hparams and \
                            hparams["type"] != default_hparams["type"]:
                        # Leave "kwargs" as-is if "type" takes value
                        # other than default.
                        parsed_hparams[name] = HParams(value, value)
                    else:
                        # Allow new hyperparameters if "type" takes default
                        # value
                        parsed_hparams[name] = HParams(
                            value, default_value, allow_new_hparam=True)
                elif name in no_typecheck_names:
                    parsed_hparams[name] = HParams(value, value)
                else:
                    parsed_hparams[name] = HParams(
                        value, default_value, allow_new_hparam)
                continue

            # Do not type-check hyperparameter named "type" and accompanied
            # with "kwargs"
            if name == "type" and "kwargs" in default_hparams:
                parsed_hparams[name] = value
                continue

            if name in no_typecheck_names:
                parsed_hparams[name] = value
            elif isinstance(value, type(default_value)):
                parsed_hparams[name] = value
            elif is_callable(value) and is_callable(default_value):
                parsed_hparams[name] = value
            else:
                try:
                    parsed_hparams[name] = type(default_value)(value)
                except TypeError:
                    raise ValueError(
                        "Hyperparameter '%s' must have type %s, got %s" %
                        (name, _type_name(default_value), _type_name(value)))

        return parsed_hparams

    @staticmethod
    def _parse_value(value, name=None):
        if isinstance(value, dict) and (name is None or name != "kwargs"):
            return HParams(value, None)
        else:
            return value

    def __getattr__(self, name):
        """Retrieves the value of the hyperparameter.
        """
        if name == '_hparams':
            return super(HParams, self).__getattribute__('_hparams')
        if name not in self._hparams:
            # Raise AttributeError to allow copy.deepcopy, etc
            raise AttributeError("Unknown hyperparameter: %s" % name)
        return self._hparams[name]

    def __getitem__(self, name):
        """Retrieves the value of the hyperparameter.
        """
        return self.__getattr__(name)

    def __setattr__(self, name, value):
        """Sets the value of the hyperparameter.
        """
        if name not in self._hparams:
            raise ValueError(
                "Unknown hyperparameter: %s. Only the `kwargs` "
                "hyperparameters can contain new entries undefined "
                "in default hyperparameters." % name)
        self._hparams[name] = self._parse_value(value, name)

    def items(self):
        """Returns the list of hyperparam `(name, value)` pairs
        """
        return iter(self)

    def keys(self):
        """Returns the list of hyperparam names
        """
        return self._hparams.keys()

    def __iter__(self):
        for name, value in self._hparams.items():
            yield name, value

    def __len__(self):
        return len(self._hparams)

    def __contains__(self, name):
        return name in self._hparams

    def __str__(self):
        """Return a string of the hparams.
        """
        hparams_dict = self.todict()
        return json.dumps(hparams_dict, sort_keys=True, indent=2)

    def get(self, name, default=None):
        """Returns the hyperparameter value for the given name. If name is not
        available then returns :attr:`default`.
        Args:
            name (str): the name of hyperparameter.
            default: the value to be returned in case name does not exist.
        """
        try:
            return self.__getattr__(name)
        except AttributeError:
            return default

    def add_hparam(self, name, value):
        """Adds a new hyperparameter.
        """
        if (name in self._hparams) or hasattr(self, name):
            raise ValueError("Hyperparameter name already exists: %s" % name)
        self._hparams[name] = self._parse_value(value, name)

    def todict(self):
        """Returns a copy of hyperparameters as a dictionary.
        """
        dict_ = copy.deepcopy(self._hparams)
        for name, value in self._hparams.items():
            if isinstance(value, HParams):
                dict_[name] = value.todict()
        return dict_