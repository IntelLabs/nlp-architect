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
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nlp_architect.nn.torch.quantization import FakeLinearQuantizationWithSTE, QuantizedLinear, \
    get_dynamic_scale, get_scale, QuantizedEmbedding


def fake_quantize_np(x, scale, bits):
    return quantize_np(x, scale, bits) / scale


def quantize_np(x, scale, bits):
    return np.clip(np.round(x * scale), -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1)


class FakeLinearQuantizationWithSTETester(unittest.TestCase):
    def test_quantization_forward(self):
        fake_quantize = FakeLinearQuantizationWithSTE().apply
        x = torch.randn(1, 100)
        scale = (2 ** (8 - 1) - 1) / np.abs(x).max()
        self.assertTrue((fake_quantize(x, scale, 8)
                         == fake_quantize_np(x, scale, 8)).all())

    def test_quantization_backward(self):
        fake_quantize = FakeLinearQuantizationWithSTE().apply
        x = torch.randn(1, 100, requires_grad=True)
        with torch.no_grad():
            scale = (2 ** (8 - 1) - 1) / x.abs().max()
        y = torch.sum(fake_quantize(x, scale, 8))
        y.backward()
        self.assertTrue((x.grad == torch.ones_like(x)).all())


class QuantizedLinearTest(unittest.TestCase):
    def test_dynamic_quantized_linear_forward(self):
        """Test QuantizedLinear forward method by giving in the input and
        weight values that are already quantized, therefore the quantization
        step should have no effect on the values and we know what values
        are expected"""
        x = torch.randn(1, 100).mul(127.).round().clamp(-127., 127.)
        qlinear = QuantizedLinear(100, 1, bias=False, requantize_output=False, mode="dynamic")
        with torch.no_grad():
            scale = 127. / qlinear.weight.abs().max()
        self.assertTrue((qlinear.fake_quantized_weight == fake_quantize_np(
            qlinear.weight.detach(), scale, 8)).all())
        qlinear.weight.data = torch.randn_like(qlinear.weight).mul(
            127.).round().clamp(-127., 127.)
        y = qlinear(x)
        self.assertEqual(y.shape, (1, 1))
        self.assertTrue((y == (x @ qlinear.weight.t())).all())

    def test_static_quantized_inference(self):
        qlinear = QuantizedLinear(
            10, 5, mode="EMA")
        weight = qlinear.weight.data.detach()
        weight_scale = get_dynamic_scale(weight, 8)
        weight_int = quantize_np(weight, weight_scale, 8)
        self.assertTrue((weight_int == torch.round(weight_int)).all())
        self.assertTrue(weight_int.abs().max() <= 127)
        x = torch.randn(3, 10) * 2 ** 0.5 - 0.36
        x_thresh = 3.
        output_thresh = 2.3
        output_scale = 127. / output_thresh
        x_scale = 127. / x_thresh
        qlinear.input_thresh = torch.tensor(x_thresh)
        qlinear.output_thresh = torch.tensor(output_thresh)
        x_int = quantize_np(x, x_scale, 8)
        self.assertTrue((x_int == torch.round(x_int)).all())
        self.assertTrue(x_int.abs().max() <= 127)
        bias = qlinear.bias.data
        bias_scale = x_scale * weight_scale
        bias_int = quantize_np(bias, bias_scale, 32)
        self.assertTrue((bias_int == torch.round(bias_int)).all())
        self.assertTrue(bias_int.abs().max() <= 2 ** (32 - 1) - 1)
        output_int = x_int @ weight_int.t() + bias_int
        output_int = torch.clamp(
            output_int, -(2 ** (32 - 1) - 1), 2 ** (32 - 1) - 1)
        output = torch.round(output_int / bias_scale
                             * output_scale).clamp(-127, 127) / output_scale
        qlinear.eval()
        qlinear_output = qlinear(x)
        self.assertTrue((qlinear_output - output).norm() < 10 ** -6)

    def test_ema_quantization(self):
        ema_decay = 0.9
        qlinear = QuantizedLinear(10, 5, bias=False, ema_decay=ema_decay, mode="EMA")
        for i in range(5):
            x = torch.randn(3, 10)
            tmp_input_thresh = x.abs().max()
            if i == 0:
                input_ema = tmp_input_thresh
            else:
                input_ema -= (1 - ema_decay) * (input_ema - tmp_input_thresh)
            y = (fake_quantize_np(x, get_scale(8, input_ema), 8)
                 @ qlinear.fake_quantized_weight.t()).detach()
            tmp_output_thresh = y.abs().max()
            if i == 0:
                output_ema = tmp_output_thresh
            else:
                output_ema -= (1 - ema_decay) * (output_ema - tmp_output_thresh)
            y = fake_quantize_np(y, get_scale(8, output_ema), 8)
            y_hat = qlinear(x)
            self.assertTrue((y == y_hat).all())
        self.assertEqual(qlinear.input_thresh, input_ema)
        self.assertEqual(qlinear.output_thresh, output_ema)

    def test_ema_quantization_data_parallel(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
            return
        ema_decay = 0.9
        fake_quantize = FakeLinearQuantizationWithSTE().apply
        qlinear = nn.DataParallel(QuantizedLinear(
            10, 5, bias=False, ema_decay=ema_decay, mode="EMA")).cuda()
        for i in range(5):
            x = torch.randn(2, 10).cuda()
            tmp_input_thresh = x[0].abs().max()
            if i == 0:
                input_ema = tmp_input_thresh
            else:
                input_ema -= (1 - ema_decay) * (input_ema - tmp_input_thresh)
            y = (fake_quantize(x, get_scale(8, input_ema), 8)
                 @ qlinear.module.fake_quantized_weight.t()).detach()
            tmp_output_thresh = y[0].abs().max()
            if i == 0:
                output_ema = tmp_output_thresh
            else:
                output_ema -= (1 - ema_decay) * \
                              (output_ema - tmp_output_thresh)
            qlinear(x)
        self.assertEqual(qlinear.module.input_thresh, input_ema)
        self.assertEqual(qlinear.module.output_thresh, output_ema)

    def test_start_quantization_delay(self):
        quantization_delay = 2
        qlinear = QuantizedLinear(10, 5, start_step=quantization_delay, mode="DYNAMIC")
        linear = nn.Linear(10, 5)
        linear.weight.data = qlinear.weight
        linear.bias.data = qlinear.bias
        for _ in range(quantization_delay):
            x = torch.randn(3, 10)
            qy = qlinear(x)
            y = linear(x)
            self.assertTrue((y == qy).all())
        qy = qlinear(x)
        self.assertFalse((y == qy).all())

    def test_start_quantization_delay_data_parallel(self):
        if not torch.cuda.is_available():
            return
        quantization_delay = 2
        qlinear = QuantizedLinear(
            10, 5, start_step=quantization_delay, mode="DYNAMIC")
        linear = nn.Linear(10, 5)
        linear.weight.data = qlinear.weight
        linear.bias.data = qlinear.bias
        qlinear = nn.DataParallel(qlinear).cuda()
        linear = nn.DataParallel(linear).cuda()
        for _ in range(quantization_delay):
            x = torch.randn(3, 10).cuda()
            qy = qlinear(x)
            y = linear(x)
            self.assertTrue((y == qy).all())
        qy = qlinear(x)
        self.assertFalse((y == qy).all())

    def test_dynamic_quantized_linear_backward(self):
        x = torch.randn(1, 100, requires_grad=True)
        linear = QuantizedLinear(100, 1, bias=False, mode="DYNAMIC")
        y = linear(x)
        y.backward()
        self.assertTrue((x.grad == linear.fake_quantized_weight).all())
        with torch.no_grad():
            scale = (2 ** (8 - 1) - 1) / x.abs().max()
        self.assertTrue((fake_quantize_np(x.detach(), scale, 8) == linear.weight.grad).all())

    def test_training_and_inference_differences_ema(self):
        qlinear = QuantizedLinear(10, 5, mode="EMA", bias=False)
        x = torch.randn(3, 10) * 2 + 0.1
        y = qlinear(x)
        qlinear.eval()
        y_hat = qlinear(x)
        self.assertTrue((y - y_hat).norm() < 1e-6)

    def test_training_and_inference_differences_dynamic(self):
        qlinear = QuantizedLinear(10, 5, bias=False)
        x = torch.randn(3, 10) * 2 + 0.1
        y = qlinear(x)
        qlinear.eval()
        y_hat = qlinear(x)
        self.assertTrue((y - y_hat).norm() < 1e-6)

    def test_none_quantized_linear(self):
        qlinear = QuantizedLinear(10, 5, mode="NONE")
        linear = nn.Linear(10, 5)
        linear.weight.data = qlinear.weight
        linear.bias.data = qlinear.bias
        x = torch.randn(3, 10)
        y = linear(x)
        y_hat = qlinear(x)
        self.assertTrue((y - y_hat).norm() < 1e-6)

    def test_export_to_8bit_with_bias(self):
        qlinear = QuantizedLinear(10, 5, mode='EMA')
        qlinear.eval()
        state_dict = qlinear.state_dict()
        self.assertTrue('weight' in state_dict)
        self.assertTrue('bias' in state_dict)
        self.assertTrue('quantized_weight' not in state_dict)
        self.assertTrue('_quantized_bias' not in state_dict)
        self.assertTrue('bias_scale' not in state_dict)
        qlinear.mode_8bit = True
        state_dict = qlinear.state_dict()
        self.assertTrue('weight' not in state_dict)
        self.assertTrue('bias' not in state_dict)
        self.assertTrue('quantized_weight' in state_dict)
        self.assertTrue(state_dict['quantized_weight'].dtype == torch.int8)
        self.assertTrue('_quantized_bias' in state_dict)
        self.assertTrue(state_dict['_quantized_bias'].dtype == torch.int32)
        self.assertTrue('bias_scale' in state_dict)
        qlinear.mode_8bit = False
        state_dict = qlinear.state_dict()
        self.assertTrue('weight' in state_dict)
        self.assertTrue('bias' in state_dict)
        self.assertTrue('quantized_weight' not in state_dict)
        self.assertTrue('_quantized_bias' not in state_dict)
        self.assertTrue('bias_scale' not in state_dict)

    def test_export_to_8bit_without_bias(self):
        qlinear = QuantizedLinear(10, 5, bias=False, mode='EMA')
        qlinear.eval()
        qlinear.mode_8bit = True
        state_dict = qlinear.state_dict()
        self.assertTrue('weight' not in state_dict)
        self.assertTrue('bias' not in state_dict)
        self.assertTrue('quantized_weight' in state_dict)
        self.assertTrue(state_dict['quantized_weight'].dtype == torch.int8)
        self.assertTrue('_quantized_bias' not in state_dict)
        self.assertTrue('bias_scale' not in state_dict)
        qlinear.mode_8bit = False
        state_dict = qlinear.state_dict()
        self.assertTrue('weight' in state_dict)
        self.assertTrue('bias' not in state_dict)
        self.assertTrue('quantized_weight' not in state_dict)
        self.assertTrue('_quantized_bias' not in state_dict)
        self.assertTrue('bias_scale' not in state_dict)

    def test_import_from_8bit_without_bias(self):
        exporter = QuantizedLinear(10, 5, bias=False, mode='dynamic')
        exporter.eval()
        exporter.mode_8bit = True
        state_dict = exporter.state_dict()
        exporter.mode_8bit = False
        importer = QuantizedLinear(10, 5, bias=False, mode='dynamic')
        self.assertTrue((exporter.weight != importer.weight).any())
        importer.eval()
        importer.load_state_dict(state_dict, strict=False)
        x = torch.randn(3, 10)
        self.assertTrue((exporter(x) == importer(x)).all())

    def test_import_from_8bit_with_bias(self):
        # QuantizationMode dynamic
        exporter = QuantizedLinear(10, 5, mode='dynamic')
        exporter.eval()
        exporter.mode_8bit = True
        state_dict = exporter.state_dict()
        exporter.mode_8bit = False
        importer = QuantizedLinear(10, 5, mode='dynamic')
        self.assertTrue((exporter.weight != importer.weight).any())
        self.assertTrue((exporter.bias != importer.bias).any())
        importer.eval()
        importer.load_state_dict(state_dict, strict=False)
        x = torch.randn(3, 10)
        self.assertTrue((exporter(x) == importer(x)).all())
        # QuantizationMode ema
        exporter = QuantizedLinear(10, 5, requantize_output=False, mode='ema')
        x = torch.randn(3, 10)
        exporter(x)
        self.assertTrue(exporter.input_thresh != 0.)
        exporter.eval()
        exporter.mode_8bit = True
        state_dict = exporter.state_dict()
        exporter.mode_8bit = False
        importer = QuantizedLinear(10, 5, requantize_output=False, mode='ema')
        self.assertTrue((exporter.weight != importer.weight).any())
        self.assertTrue((exporter.bias != importer.bias).any())
        importer.eval()
        importer.load_state_dict(state_dict, strict=False)
        self.assertTrue((exporter(x) == importer(x)).all())

    def test_train_block_when_loading_quantized_model(self):
        exporter = QuantizedLinear(10, 5, mode='dynamic')
        exporter.eval()
        exporter.mode_8bit = True
        state_dict = exporter.state_dict()
        importer = QuantizedLinear(10, 5, mode='dynamic')
        importer.eval()
        importer.load_state_dict(state_dict, strict=False)
        with self.assertRaises(RuntimeError):
            importer.train()

    def test_restrict_loading_to_train_model(self):
        exporter = QuantizedLinear(10, 5, mode='dynamic')
        exporter.eval()
        exporter.mode_8bit = True
        state_dict = exporter.state_dict()
        importer = QuantizedLinear(10, 5, mode='dynamic')
        with self.assertRaises(RuntimeError):
            importer.load_state_dict(state_dict, strict=False)


class QuantizedEmbeddingTest(unittest.TestCase):
    def test_quantized_embedding_training_forward(self):
        embedding = QuantizedEmbedding(10, 3, mode="ema")
        with torch.no_grad():
            scale = 127. / embedding.weight.abs().max()
        self.assertTrue((embedding.fake_quantized_weight == fake_quantize_np(
            embedding.weight.detach(), scale, 8)).all())
        embedding.weight.data = torch.randn_like(embedding.weight).mul(
            127.).round().clamp(-127., 127.)
        indices = torch.tensor(np.arange(10))
        ground = F.embedding(indices, embedding.weight)
        quantized = embedding(indices)
        self.assertTrue((ground == quantized).all())

    def test_quantized_embedding_inference_forward(self):
        embedding = QuantizedEmbedding(10, 3, mode="ema")
        with torch.no_grad():
            scale = 127. / embedding.weight.abs().max()
        self.assertTrue((embedding.fake_quantized_weight == fake_quantize_np(
            embedding.weight.detach(), scale, 8)).all())
        embedding.weight.data = torch.randn_like(embedding.weight).mul(
            127.).round().clamp(-127., 127.)
        indices = torch.tensor(np.arange(10))
        embedding.eval()
        ground = F.embedding(indices, embedding.weight)
        quantized = embedding(indices)
        self.assertTrue((ground == quantized).all())

    def test_quantized_embedding_backward(self):
        embedding = QuantizedEmbedding(10, 3, mode="ema")
        linear = nn.Linear(3, 1)
        indices = torch.tensor([2])
        h = embedding(indices)
        y = linear(h)
        y.backward()
        grad = torch.zeros_like(embedding.weight)
        grad[indices.item(), :] = linear.weight.t().squeeze()
        self.assertTrue((embedding.weight.grad == grad).all())
        self.assertTrue((linear.weight.grad == h).all())

    def test_delay_quantization_start(self):
        qembedding = QuantizedEmbedding(10, 3, mode="ema", start_step=1)
        embedding = nn.Embedding(10, 3)
        embedding.weight.data = qembedding.weight
        indices = torch.tensor(np.arange(10))
        self.assertTrue((embedding(indices) == qembedding(indices)).all())
        self.assertTrue((embedding(indices) != qembedding(indices)).any())

    def test_quantization_turned_off(self):
        qembedding = QuantizedEmbedding(10, 3, mode="none")
        embedding = nn.Embedding(10, 3)
        embedding.weight.data = qembedding.weight
        indices = torch.tensor(np.arange(10))
        self.assertTrue((embedding(indices) == qembedding(indices)).all())
        self.assertTrue((embedding(indices) == qembedding(indices)).all())

    def test_export_to_8bit(self):
        qembed = QuantizedEmbedding(10, 5, mode='EMA')
        qembed.eval()
        state_dict = qembed.state_dict()
        self.assertTrue('quantized_weight' not in state_dict)
        self.assertTrue('weight' in state_dict)
        qembed.mode_8bit = True
        state_dict = qembed.state_dict()
        self.assertTrue('quantized_weight' in state_dict)
        self.assertTrue(state_dict['quantized_weight'].dtype == torch.int8)
        self.assertTrue('weight' not in state_dict)
        qembed.mode_8bit = False
        state_dict = qembed.state_dict()
        self.assertTrue('quantized_weight' not in state_dict)
        self.assertTrue('weight' in state_dict)

    def test_load_from_8bit(self):
        exporter = QuantizedEmbedding(10, 5, mode='EMA')
        exporter.eval()
        exporter.mode_8bit = True
        state_dict = exporter.state_dict()
        exporter.mode_8bit = False
        importer = QuantizedEmbedding(10, 5, mode='EMA')
        self.assertTrue((exporter.weight != importer.weight).any())
        importer.eval()
        importer.load_state_dict(state_dict, strict=False)
        indices = torch.tensor(np.arange(10))
        self.assertTrue((exporter(indices) == importer(indices)).all())
