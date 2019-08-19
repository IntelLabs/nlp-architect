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

import torch.nn.functional as F

from nlp_architect.models import TrainableModel


class TeacherStudentDistill:
    def __init__(self, teacher_model: TrainableModel,
                 temperature: float = 1.0,
                 kd_w: float = 0.5,
                 loss_w: float = 0.5):
        """
        Teacher-Student knowledge distillation helper.
        Use this object when training a model with KD and a teacher model.
        
        Args:
            teacher_model (TrainableModel): teacher model
            temperature (float, optional): KD temperature. Defaults to 1.0.
            kd_w (float, optional): teacher loss weight. Defaults to 0.5.
            loss_w (float, optional): student loss weight. Defaults to 0.5.
        """
        self.teacher = teacher_model
        self.t = temperature
        self.kd_w = kd_w
        self.loss_w = loss_w

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
        parser.add_argument("--kd_t", type=float, default=1.0,
                            help="KD temperature")
        parser.add_argument("--kd_teacher_w", type=float, default=0.5,
                            help="KD teacher loss weight")
        parser.add_argument("--kd_student_w", type=float, default=0.5,
                            help="KD student loss weight")

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
        student_log_sm = F.log_softmax(student_logits / self.t, dim=2)
        teacher_log_sm = F.softmax(teacher_logits / self.t, dim=2)
        distill_loss = F.mse_loss(student_log_sm, teacher_log_sm.detach())
        # distill_loss = F.kl_div(student_log_sm,
        #                         teacher_log_sm.detach(),
        #                         size_average=False) / teacher_log_sm.shape[0]
        # normalize losses ?!? 
        return loss * self.loss_w + distill_loss * self.kd_w
