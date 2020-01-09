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
import argparse
import logging

import torch.nn as nn
import torch.nn.functional as F

from nlp_architect.models import TrainableModel

logger = logging.getLogger(__name__)


MSE_loss = nn.MSELoss(reduction="mean")
KL_loss = nn.KLDivLoss(reduction="batchmean")

losses = {
    "kl": KL_loss,
    "mse": MSE_loss,
}

TEACHER_TYPES = ["bert"]


class TeacherStudentDistill:
    """
        Teacher-Student knowledge distillation helper.
        Use this object when training a model with KD and a teacher model.

        Args:
            teacher_model (TrainableModel): teacher model
            temperature (float, optional): KD temperature. Defaults to 1.0.
            dist_w (float, optional): distillation loss weight. Defaults to 0.1.
            loss_w (float, optional): student loss weight. Defaults to 1.0.
            loss_function (str, optional): loss function to use (kl for KLDivLoss,
                mse for MSELoss)
        """

    def __init__(
        self,
        teacher_model: TrainableModel,
        temperature: float = 1.0,
        dist_w: float = 0.1,
        loss_w: float = 1.0,
        loss_function="kl",
    ):
        self.teacher = teacher_model
        self.t = temperature
        self.dist_w = dist_w
        self.loss_w = loss_w
        self.loss_fn = losses.get(loss_function, KL_loss)

    def get_teacher_logits(self, inputs):
        """
        Get teacher logits

        Args:
            inputs: input

        Returns:
            teachr logits
        """
        return self.teacher.get_logits(inputs)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """
        Add KD arguments to parser

        Args:
            parser (argparse.ArgumentParser): parser
        """
        parser.add_argument(
            "--teacher_model_path", type=str, required=True, help="Path to teacher model"
        )
        parser.add_argument(
            "--teacher_model_type",
            type=str,
            required=True,
            choices=TEACHER_TYPES,
            help="Teacher model class type",
        )
        parser.add_argument("--kd_temp", type=float, default=1.0, help="KD temperature value")
        parser.add_argument(
            "--kd_loss_fn", type=str, choices=["kl", "mse"], default="mse", help="KD loss function"
        )
        parser.add_argument("--kd_dist_w", type=float, default=0.1, help="KD weight on loss")
        parser.add_argument(
            "--kd_student_w", type=float, default=1.0, help="KD student weight on loss"
        )

    def distill_loss(self, loss, student_logits, teacher_logits):
        """
        Add KD loss

        Args:
            loss: student loss
            student_logits: student model logits
            teacher_logits: teacher model logits

        Returns:
            KD loss
        """
        student_log_sm = F.log_softmax(student_logits / self.t, dim=-1)
        teacher_log_sm = F.softmax(teacher_logits / self.t, dim=-1)
        distill_loss = self.loss_fn(input=student_log_sm, target=teacher_log_sm)
        return self.loss_w * loss + distill_loss * self.dist_w * (self.t ** 2)
