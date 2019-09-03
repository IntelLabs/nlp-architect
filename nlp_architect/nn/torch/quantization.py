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
    return 2**(bits - 1) - 1


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


class QuantizedLinear(nn.Linear):
    """Linear layer with quantization aware training capability"""

    def __init__(self, *args, activation_bits=8, weight_bits=8,
                 requantize_output=True, ema_decay=0.9999, start_step=0, mode='none',
                 **kwargs):
        super().__init__(*args, **kwargs)
        if activation_bits < 2 or weight_bits < 2:
            raise ValueError(
                f"activation_bits={activation_bits} and weight_bits="
                f"{weight_bits} must be higher than 1 ")
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits
        self.accumulation_bits = 32
        self.ema_decay = ema_decay
        self.requantize_output = requantize_output
        self.start_step = start_step
        self.mode = QuantizationMode[mode.upper()]
        self.register_buffer('_step', torch.zeros(1))
        self.register_buffer('input_thresh', torch.zeros(1))
        self.register_buffer('output_thresh', torch.zeros(1))

    def training_quantized_forward(self, input):
        """fake quantized forward, fake quantizes weights and activations,
        learn quantization ranges if quantization mode is EMA.
        This function should only be used while training"""
        assert self.training, "should only be called when training"
        if self.mode == QuantizationMode.EMA:
            self._update_ema(self.input_thresh, input.detach())
        input_scale = self._get_input_scale(input)
        out = F.linear(_fake_quantize(input, input_scale, self.activation_bits),
                       self.fake_quantized_weight, self.bias)
        if self.requantize_output:
            if self.mode == QuantizationMode.EMA:
                self._update_ema(self.output_thresh, out.detach())
            out = _fake_quantize(
                out, self._get_output_scale(out), self.activation_bits)
        return out

    @classmethod
    def from_config(cls, *args, config=None, **kwargs):
        """Initialize quantized layer from config"""
        keys = ['weight_bits', 'start_step', 'mode',
                'activation_bits', 'requantize_output', 'ema_decay']
        return cls(*args, **kwargs, **{k: getattr(config, k) for k in keys})

    def inference_quantized_forward(self, input):
        """Simulate quantized inference. quantize input and perform calculation with only integer numbers.
        This function should only be used while doing inference"""
        assert not self.training, "should only be called when not training"
        input_scale = self._get_input_scale(input)
        dequantize_scale = self.weight_scale * input_scale
        quantized_input = quantize(input, input_scale, self.activation_bits)
        out = F.linear(quantized_input, self.quantized_weight,
                       self.get_quantized_bias(dequantize_scale))
        out = dequantize(out, dequantize_scale)
        if self.requantize_output:
            output_scale = self._get_output_scale(out)
            out = dequantize(quantize(out, output_scale, self.activation_bits), output_scale)
        return out

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

    def get_quantized_bias(self, scale):
        try:
            bias = quantize(self.bias, scale, self.accumulation_bits)
        except AttributeError:
            bias = None
        return bias

    @property
    def fake_quantized_weight(self):
        return _fake_quantize(self.weight, self.weight_scale, self.weight_bits)

    @property
    def quantized_weight(self):
        return quantize(self.weight, self.weight_scale, self.weight_bits)

    @property
    def weight_scale(self):
        return get_dynamic_scale(self.weight, self.weight_bits)

    def _get_input_scale(self, input):
        return self._get_activation_scale(input, self.input_thresh)

    def _get_output_scale(self, output):
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

    def extra_repr(self):
        return super().extra_repr() + f', mode={self.mode}, activation_bits=' \
            f'{self.activation_bits}, weight_bits={self.weight_bits}, ema_decay=' \
            f'{self.ema_decay if self.mode == QuantizationMode.EMA else None}'


class QuantizedEmbedding(nn.Embedding):
    """Embedding layer with quantization aware training capability"""

    def __init__(self, *args, weight_bits=8, start_step=0, mode='none', **kwargs):
        super().__init__(*args, **kwargs)
        if weight_bits < 2:
            raise ValueError(
                f"weight_bits={weight_bits} must be higher than 1 ")
        self.weight_bits = weight_bits
        self.mode = QuantizationMode[mode.upper()]
        self.start_step = start_step
        self.register_buffer('_step', torch.zeros(1))

    def forward(self, input):
        if self.mode == QuantizationMode.NONE:
            return super().forward(input)
        if self._step >= self.start_step:
            out = self.quantized_forward(input)
        else:
            out = super().forward(input)
        if self.training:
            self._step += 1
        return out

    @classmethod
    def from_config(cls, *args, config=None, **kwargs):
        """Initialize quantized layer from config"""
        keys = ['weight_bits', 'start_step', 'mode']
        return cls(*args, **kwargs, **{k: getattr(config, k) for k in keys})

    def quantized_forward(self, input):
        """Return quantized embeddings"""
        return F.embedding(
            input, self.fake_quantized_weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    @property
    def fake_quantized_weight(self):
        return _fake_quantize(self.weight, self.weight_scale, self.weight_bits)

    @property
    def weight_scale(self):
        return get_dynamic_scale(self.weight, self.weight_bits)

    def extra_repr(self):
        return super().extra_repr() + f", mode={self.mode}, weight_bits={self.weight_bits}"


class QuantizationConfig(Config):
    """Quantization Configuration Object"""
    def __init__(self,
                 activation_bits=8,
                 weight_bits=8,
                 mode='none',
                 start_step=0,
                 ema_decay=0.9999,
                 requantize_output=True
                 ):
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits
        self.mode = mode
        self.start_step = start_step
        self.ema_decay = ema_decay
        self.requantize_output = requantize_output
