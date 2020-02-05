# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
# pylint: disable=no-member
"""
Quantization ops
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F

from nlp_architect.common import Config


logger = logging.getLogger(__name__)


def get_dynamic_scale(x, bits, with_grad=False):
    """Calculate dynamic scale for quantization from input by taking the
    maximum absolute value from x and number of bits"""
    with torch.set_grad_enabled(with_grad):
        threshold = x.abs().max()
    return get_scale(bits, threshold)


def get_scale(bits, threshold):
    """Calculate scale for quantization according to some constant and number of bits"""
    return calc_max_quant_value(bits) / threshold


def calc_max_quant_value(bits):
    """Calculate the maximum symmetric quantized value according to number of bits"""
    return 2 ** (bits - 1) - 1


def quantize(input, scale, bits):
    """Do linear quantization to input according to a scale and number of bits"""
    thresh = calc_max_quant_value(bits)
    return input.mul(scale).round().clamp(-thresh, thresh)


def dequantize(input, scale):
    """linear dequantization according to some scale"""
    return input.div(scale)


# TODO(ofir) future work, implement a layer that uses this function that gives a more comfortable
class FakeLinearQuantizationWithSTE(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop"""

    @staticmethod
    def forward(ctx, input, scale, bits=8):
        """fake quantize input according to scale and number of bits, dequantize
        quantize(input))"""
        return dequantize(quantize(input, scale, bits), scale)

    @staticmethod
    def backward(ctx, grad_output):
        """Calculate estimated gradients for fake quantization using
        Straight-Through Estimator (STE) according to:
        https://openreview.net/pdf?id=B1ae1lZRb"""
        return grad_output, None, None


class QuantizationMode(Enum):
    NONE = auto()
    DYNAMIC = auto()
    EMA = auto()


_fake_quantize = FakeLinearQuantizationWithSTE.apply


class QuantizedLayer(ABC):
    """Quantized Layer interface"""

    CONFIG_ATTRIBUTES = ["weight_bits", "start_step", "mode"]
    REPR_ATTRIBUTES = ["mode", "weight_bits"]

    def __init__(self, *args, weight_bits=8, start_step=0, mode="none", **kwargs):
        if weight_bits < 2:
            raise ValueError(f"weight_bits={weight_bits} must be higher than 1 ")
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.mode = QuantizationMode[mode.upper()]
        self.start_step = start_step
        self.register_buffer("_step", torch.zeros(1))
        # buffers for inference
        self.register_buffer("quantized_weight", None)
        self.register_buffer("_weight_scale", None)
        # handle import and export in 8bit
        self.mode_8bit = False
        self._imported_from_quantized = False
        # register saving hook
        self._register_state_dict_hook(self._state_dict_hook)

    def forward(self, input):
        if self.mode == QuantizationMode.NONE:
            return super().forward(input)
        if self.training:
            if self._step >= self.start_step:
                out = self.training_quantized_forward(input)
            else:
                out = super().forward(input)
            self._step += 1
        else:
            out = self.inference_quantized_forward(input)
        return out

    @abstractmethod
    def training_quantized_forward(self, input):
        """Implement forward method to be used while training"""

    @abstractmethod
    def inference_quantized_forward(self, input):
        """Implement forward method to be used while evaluating"""

    @classmethod
    def from_config(cls, *args, config=None, **kwargs):
        """Initialize quantized layer from config"""
        return cls(*args, **kwargs, **{k: getattr(config, k) for k in cls.CONFIG_ATTRIBUTES})

    @property
    def fake_quantized_weight(self):
        return _fake_quantize(self.weight, self.weight_scale, self.weight_bits)

    @property
    def weight_scale(self):
        return (
            get_dynamic_scale(self.weight, self.weight_bits)
            if self.training
            else self._weight_scale
        )

    def train(self, mode=True):
        """handle transition between quantized model and simulated quantization"""
        if self.training != mode:
            if mode:
                if self._imported_from_quantized:
                    raise RuntimeError(
                        "Model imported from quantized checkpoint cannot be moved to \
                            training mode"
                    )
                self._train()
            else:
                self._eval()
        super().train(mode)

    def _train(self):
        """function to be called by self.train(mode=True) which modifies modules attributes\
             according to the model"""

    def _eval(self):
        """function to be called by self.train(mode=False), or eval() which modifies modules\
             attributes according to the model"""
        self._weight_scale = self.weight_scale
        self.quantized_weight = quantize(self.weight, self.weight_scale, self.weight_bits)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """check if model is loaded from quantized checkpoint or regular checkpoint"""
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if state_dict.get(prefix + "quantized_weight", None) is not None:
            if self.training:
                raise RuntimeError(
                    "Can't load quantized model in training mode, first change model's \
                         to evaluation and then load the saved model"
                )
            self._imported_from_quantized = True

    @staticmethod
    def _state_dict_hook(module, state_dict, prefix, local_metadata):
        """hook to be registered to module when exporting the model to 8bit, can be overrided\
             to customize to layer behaviour"""
        if module.mode_8bit and module.mode != QuantizationMode.NONE:
            state_dict.pop(prefix + "weight", None)
            state_dict.pop(prefix + "_step", None)
            state_dict[prefix + "quantized_weight"] = state_dict[prefix + "quantized_weight"].char()
        else:
            state_dict.pop(prefix + "quantized_weight", None)
            state_dict.pop(prefix + "_weight_scale", None)

    def extra_repr(self):
        s = ""
        for entry in self.REPR_ATTRIBUTES:
            s += f", {entry}={getattr(self, entry)}"
        return super().extra_repr() + s


class QuantizedLinear(QuantizedLayer, nn.Linear):
    """Linear layer with quantization aware training capability"""

    CONFIG_ATTRIBUTES = QuantizedLayer.CONFIG_ATTRIBUTES + [
        "activation_bits",
        "requantize_output",
        "ema_decay",
    ]
    REPR_ATTRIBUTES = QuantizedLayer.REPR_ATTRIBUTES + [
        "activation_bits",
        "accumulation_bits",
        "ema_decay",
        "requantize_output",
    ]

    def __init__(
        self, *args, activation_bits=8, requantize_output=True, ema_decay=0.9999, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if activation_bits < 2:
            raise ValueError(f"activation_bits={activation_bits} must be higher than 1 ")
        self.activation_bits = activation_bits
        self.accumulation_bits = 32
        self.ema_decay = ema_decay
        self.requantize_output = requantize_output
        self.register_buffer("input_thresh", torch.zeros(1))
        if self.requantize_output:
            self.register_buffer("output_thresh", torch.zeros(1))
        # real quantization
        if kwargs.get("bias", True):
            self.register_buffer("_quantized_bias", None)
            self.register_buffer("bias_scale", None)

    def training_quantized_forward(self, input):
        """fake quantized forward, fake quantizes weights and activations,
        learn quantization ranges if quantization mode is EMA.
        This function should only be used while training"""
        assert self.training, "should only be called when training"
        if self.mode == QuantizationMode.EMA:
            self._update_ema(self.input_thresh, input.detach())
        input_scale = self._get_input_scale(input)
        out = F.linear(
            _fake_quantize(input, input_scale, self.activation_bits),
            self.fake_quantized_weight,
            self.bias,
        )
        if self.requantize_output:
            if self.mode == QuantizationMode.EMA:
                self._update_ema(self.output_thresh, out.detach())
            out = _fake_quantize(out, self._get_output_scale(out), self.activation_bits)
        return out

    def inference_quantized_forward(self, input):
        """Simulate quantized inference. quantize input and perform calculation with only integer numbers.
        This function should only be used while doing inference"""
        assert not self.training, "should only be called when not training"
        input_scale = self._get_input_scale(input)
        self.bias_scale = self.weight_scale * input_scale
        quantized_input = quantize(input, input_scale, self.activation_bits)
        out = F.linear(quantized_input, self.quantized_weight, self.quantized_bias)
        # TODO(ofir) fuse the operation of requantization with dequantiz
        out = dequantize(out, self.bias_scale)
        if self.requantize_output:
            output_scale = self._get_output_scale(out)
            out = dequantize(quantize(out, output_scale, self.activation_bits), output_scale)
        return out

    def _eval(self):
        super()._eval()
        if self.mode == QuantizationMode.EMA and self.bias is not None:
            self.bias_scale = self._get_input_scale() * self.weight_scale
            self.quantized_bias = quantize(self.bias, self.bias_scale, self.accumulation_bits)

    @staticmethod
    def _state_dict_hook(module, state_dict, prefix, local_metadata):
        """hook to be registered to module when exporting the model to 8bit,\
             can be overrided to customize to layer behaviour"""
        super()._state_dict_hook(module, state_dict, prefix, local_metadata)
        if module.mode_8bit:
            if module.mode == QuantizationMode.EMA:
                state_dict.pop(prefix + "bias", None)
                try:
                    state_dict[prefix + "_quantized_bias"] = state_dict[
                        prefix + "_quantized_bias"
                    ].int()
                except KeyError:
                    # in case there is no bias dont do anything
                    pass
        else:
            state_dict.pop(prefix + "_quantized_bias", None)
            state_dict.pop(prefix + "bias_scale", None)

    @property
    def quantized_bias(self):
        try:
            if self.mode == QuantizationMode.EMA:
                bias = self._quantized_bias
            elif self.mode == QuantizationMode.DYNAMIC:
                bias = quantize(self.bias, self.bias_scale, self.accumulation_bits)
            else:
                raise RuntimeError(f"Unknown quantization mode: {self.mode}")
        except AttributeError:
            bias = None
        return bias

    @quantized_bias.setter
    def quantized_bias(self, value):
        self._quantized_bias = value

    def _get_input_scale(self, input=None):
        return self._get_activation_scale(input, self.input_thresh)

    def _get_output_scale(self, output=None):
        return self._get_activation_scale(output, self.output_thresh)

    def _get_activation_scale(self, activation, threshold):
        if self.mode == QuantizationMode.DYNAMIC:
            scale = get_dynamic_scale(activation, self.activation_bits)
        elif self.mode == QuantizationMode.EMA:
            scale = get_scale(self.activation_bits, threshold)
        return scale

    def _update_ema(self, ema, input, reduce_fn=lambda x: x.abs().max()):
        """Update exponential moving average (EMA) of activations thresholds.
        the reduce_fn calculates the current threshold from the input tensor"""
        assert self._step >= self.start_step
        if self._step == self.start_step:
            ema.fill_(reduce_fn(input))
        else:
            ema.sub_((1 - self.ema_decay) * (ema - reduce_fn(input)))


class QuantizedEmbedding(QuantizedLayer, nn.Embedding):
    """Embedding layer with quantization aware training capability"""

    def training_quantized_forward(self, input):
        """Return quantized embeddings"""
        assert self.training, "should only be called when training"
        return F.embedding(
            input,
            self.fake_quantized_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def inference_quantized_forward(self, input):
        """forward to be used during inference"""
        assert not self.training, "should only be called when not training"
        q_embeddings = F.embedding(
            input,
            self.quantized_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return dequantize(q_embeddings, self.weight_scale)


class QuantizationConfig(Config):
    """Quantization Configuration Object"""

    ATTRIBUTES = {
        "activation_bits": 8,
        "weight_bits": 8,
        "mode": "none",
        "start_step": 0,
        "ema_decay": 0.9999,
        "requantize_output": True,
    }
