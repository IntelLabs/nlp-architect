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
import io
import logging
import os
import pickle
from typing import List
import numpy as np
import torch
from torch import nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from nlp_architect.data.sequence_classification import SequenceClsInputExample
from nlp_architect.models import TrainableModel
from nlp_architect.nn.torch.layers import CRF
from nlp_architect.nn.torch.distillation import TeacherStudentDistill
from nlp_architect.nn.torch.modules.embedders import IDCNN
from nlp_architect.utils.metrics import tagging
from nlp_architect.utils.text import Vocabulary, char_to_id
from nlp_architect.utils.metrics import accuracy

logger = logging.getLogger(__name__)


class CNN(nn.Module):

    def __init__(self,
            vocab_size: int,
            num_labels: int,
            max_seq_len: int = 15,
            embedding_dim: int = 50,
            cnn_num_filters: int = 50,
            dropout: float = 0.5,
            padding_idx: int = 0,
        ):
        super(CNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # self.pretrained_embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze=True)
        #self.lstm_layer = nn.LSTM(embedding_dim,50,num_layers=1,batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        self.con1 = nn.Conv1d(embedding_dim, cnn_num_filters, kernel_size=3)
        self.maxp1 = nn.MaxPool1d(max_seq_len -2 ) # beacuse kernel size is 3
        
        self.con2 = nn.Conv1d(embedding_dim, cnn_num_filters, kernel_size=4)
        self.maxp2 = nn.MaxPool1d(max_seq_len - 3)
        
        self.con3 = nn.Conv1d(embedding_dim, cnn_num_filters, kernel_size=5)
        self.maxp3 = nn.MaxPool1d(max_seq_len - 4)
        
        self.fc = nn.Linear(3 * cnn_num_filters, num_labels)

    def load_embeddings(self, embeddings):
        """
        Load pre-defined word embeddings

        Args:
            embeddings (torch.tensor): word embedding tensor
        """
        self.word_embeddings = nn.Embedding.from_pretrained(
            embeddings, freeze=False, padding_idx=self.padding_idx
        )

    def forward(self, words, **kwargs):
        x = self.word_embeddings(words) # shape -> bs x T x d
        x = x.transpose(1,2) # shape -> bs x d x T
        
        x1=F.relu(self.con1(x))
        x1=self.maxp1(x1)
        x1=x1.flatten(start_dim=1)
        #x1=x1.reshape(x1.shape[0],-1)

        x2=F.relu(self.con2(x))
        x2=self.maxp2(x2)
        x2=x2.flatten(start_dim=1)
        
        x3=F.relu(self.con3(x))
        x3=self.maxp3(x3)
        x3=x3.flatten(start_dim=1)
        
        x=torch.cat([x1,x2,x3],dim=1)
        x=self.dropout(x)
        x=self.fc(x)
        y=torch.sigmoid(x)
        #y1,y2=self.lstm_layer(x,None)
        return y


class GLUE(TrainableModel):
    """
    A model for training on glue tasks
    Supports multi-gpu training, KD from teacher models

    """
    def __init__(
        self,
        embedder_model,
        word_vocab: Vocabulary,
        labels: List[str] = None,
        device: str = "cpu",
        n_gpus=0,
        task_type="classification",
        metric_fn=accuracy,
    ):
        super(GLUE, self).__init__()
        self.model = embedder_model
        self.labels = labels
        self.label_str_id = {l: i for i, l in enumerate(self.labels)}
        self.label_id_str = {v: k for k, v in self.label_str_id.items()}
        self.word_vocab = word_vocab
        self.device = device
        self.n_gpus = n_gpus
        self.task_type = task_type
        self.metric_fn = metric_fn
        self.to(self.device, self.n_gpus)

    def convert_to_tensors(
        self,
        examples: List[SequenceClsInputExample],
        max_seq_length: int = 128,
        pad_id: int = 0,
        labels_pad_id: int = 0,
        include_labels: bool = True,
    ) -> TensorDataset:
        """
        Convert examples to valid sequence classification dataset


        Returns:
            TensorDataset: TensorDataset for given examples
        """

        features = []
        for example in examples:
            word_tokens = [self.word_vocab[t] for t in example.tokens]
            word_tokens_b = None
            mask_b = None
            if example.tokens_b is not None:
                word_tokens_b = [self.word_vocab[t] for t in example.tokens_b]
            label = None
            if include_labels:
                label = self.label_str_id.get(example.label)

            # cut up to max length
            word_tokens = word_tokens[:max_seq_length]
            if word_tokens_b is not None:
                word_tokens_b = word_tokens_b[:max_seq_length]
                mask_b = [1] * len(word_tokens_b)
            mask = [1] * len(word_tokens)
            

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(word_tokens)
            input_ids = word_tokens + ([pad_id] * padding_length)
            mask = mask + ([0] * padding_length)
            assert len(input_ids) == max_seq_length
            input_ids_b = None
            mask_b = None
            if word_tokens_b is not None:
                padding_length_b = max_seq_length - len(word_tokens_b)
                input_ids_b = word_tokens + ([pad_id] * padding_length_b)
                mask_b = mask_b + ([0] * padding_length_b)
                assert len(input_ids_b) == max_seq_length

            features.append(
                InputFeatures(
                    input_ids,
                    input_ids_b=input_ids_b,
                    mask=mask,
                    mask_b=mask_b,
                    label_id=label if include_labels else None,
                )
            )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)

        # all_input_ids_b = torch.tensor([f.input_ids_b for f in features], dtype=torch.long)
        masks = torch.tensor([f.mask for f in features], dtype=torch.long)
        # masks_b = torch.tensor([f.mask_b for f in features], dtype=torch.long)
        if include_labels:
            is_labeled = torch.tensor([True for f in features], dtype=torch.bool)
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, masks, is_labeled, all_label_ids
            )
        else:
            is_labeled = torch.tensor([False for f in features], dtype=torch.bool)
            dataset = TensorDataset(all_input_ids, masks, is_labeled)
        return dataset


    def _postprocess_logits(self, logits):
        # preds = logits.numpy()
        preds = logits
        if self.task_type == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.task_type == "regression":
            preds = np.squeeze(preds)
        return preds

        
    def get_optimizer(self, opt_fn=None, lr: int = 0.001):
        """
        Get default optimizer

        Args:
            lr (int, optional): learning rate. Defaults to 0.001.

        Returns:
            torch.optim.Optimizer: optimizer
        """
        params = self.model.parameters()
        if opt_fn is None:
            opt_fn = optim.Adam
        return opt_fn(params, lr=lr)

    @staticmethod
    def batch_mapper(batch):
        """
        Map batch to correct input names
        """
        mapping = {"words": batch[0], "mask": batch[1], "is_labeled": batch[2]}
        if len(batch) == 4:
            mapping.update({"labels": batch[3]})
        return mapping


    def train(self, train_data_set: DataLoader,
              dev_data_set: DataLoader = None,
              epochs: int = 3,
              batch_size: int = 8,
              optimizer=None,
              max_grad_norm: float = 5.0,
              logging_steps: int = 50,
              save_steps: int = 100,
              save_path: str = None,
              distiller: TeacherStudentDistill = None,
              best_result_file: str = None,
              word_dropout: float = 0):
        """
        Train a tagging model

        Args:
            train_data_set (DataLoader): train examples dataloader.
                - If distiller object is provided train examples should contain a tuple of
                  student/teacher data examples.
            dev_data_set (DataLoader, optional): dev examples dataloader. Defaults to None.
            test_data_set (DataLoader, optional): test examples dataloader. Defaults to None.
            epochs (int, optional): num of epochs to train. Defaults to 3.
            batch_size (int, optional): batch size. Defaults to 8.
            optimizer (fn, optional): optimizer function. Defaults to default model optimizer.
            max_grad_norm (float, optional): max gradient norm. Defaults to 5.0.
            logging_steps (int, optional): number of steps between logging. Defaults to 50.
            save_steps (int, optional): number of steps between model saves. Defaults to 100.
            save_path (str, optional): model output path. Defaults to None.
            distiller (TeacherStudentDistill, optional): KD model for training the model using
            a teacher model. Defaults to None.
            best_result_file (str, optional): path to save best dev results when it's updated.
            word_dropout (float, optional): whole-word (-> oov) dropout rate. Defaults to 0.
        """
        if optimizer is None:
            optimizer = self.get_optimizer()
        train_batch_size = batch_size * max(1, self.n_gpus)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data_set.dataset))
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Instantaneous batch size per GPU/CPU = %d",
                    batch_size)
        logger.info("  Total batch size = %d", train_batch_size)
        global_step = 0
        best_dev = 0
        dev_test = 0
        self.model.zero_grad()
        epoch_it = trange(epochs, desc="Epoch")
        for epoch in epoch_it:
            step_it = tqdm(train_data_set, desc="Train iteration")
            avg_loss = 0
            
            for step, batches in enumerate(step_it):
                self.model.train()

                batch, t_batch = (batches, []) if not distiller else (batches[:2])
                batch = tuple(t.to(self.device) for t in batch)
                inputs = self.batch_mapper(batch)
                logits = self.model(**inputs)

                if distiller:
                    t_batch = tuple(t.to(self.device) for t in t_batch)
                    t_logits = distiller.get_teacher_logits(t_batch)
                    t_labels = torch.argmax(F.log_softmax(t_logits, dim=-1), dim=-1)
                
                # pseudo labeling
                for i, is_labeled in enumerate(inputs['is_labeled']):
                    if not is_labeled:
                        inputs['labels'][i] = t_labels[i]
                
               
                # loss
                loss_fn = CrossEntropyLoss()  # CrossEntropyLoss(ignore_index=0)
                loss = loss_fn(logits.view(-1, len(self.labels)), inputs['labels'].view(-1))
                    
                if self.n_gpus > 1:
                    loss = loss.mean()

                # # add distillation loss if activated
                # if distiller:
                #     loss = distiller.distill_loss(loss, logits, t_logits)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                avg_loss += loss.item()
                if global_step % logging_steps == 0:
                    if step != 0:
                        logger.info(
                            " global_step = %s, average loss = %s", global_step, avg_loss / step)
                        best_dev = self.update_best_model(dev_data_set, best_dev, 
                                                    best_result_file, avg_loss / step, epoch, save_path=None)
                if save_steps != 0 and save_path is not None and \
                        global_step % save_steps == 0:
                    self.save_model(save_path)
        self.update_best_model(dev_data_set, best_dev, best_result_file, 'end_training', 'end_training', save_path=save_path + '/best_dev')


    def to(self, device="cpu", n_gpus=0):
        """
        Put model on given device

        Args:
            device (str, optional): device backend. Defaults to 'cpu'.
            n_gpus (int, optional): number of gpus. Defaults to 0.
        """
        if self.model is not None:
            self.model.to(device)
            if n_gpus > 1:
                self.model = torch.nn.DataParallel(self.model)
        self.device = device
        self.n_gpus = n_gpus

    def evaluate(self, data_set: DataLoader):
        """
        Run evaluation on given dataloader

        Args:
            data_set (DataLoader): a data loader to run evaluation on

        Returns:
            logits, labels (if labels are given)
        """
        logger.info("***** Running inference *****")
        logger.info(" Batch size: {}".format(data_set.batch_size))
        preds = None
        out_label_ids = None
        for batch in tqdm(data_set, desc="Inference iteration"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = self.batch_mapper(batch)
                logits = self.model(**inputs)
            model_output = logits.detach().cpu()
            model_out_label_ids = inputs["labels"].detach().cpu() if "labels" in inputs else None
            if preds is None:
                preds = model_output
                out_label_ids = model_out_label_ids
            else:
                preds = torch.cat((preds, model_output), dim=0)
                out_label_ids = (
                    torch.cat((out_label_ids, model_out_label_ids), dim=0)
                    if out_label_ids is not None
                    else None
                )
        output = (preds,)
        if out_label_ids is not None:
            output = output + (out_label_ids,)
        return output

    def evaluate_predictions(self, logits, label_ids):
        """
        Run evaluation of given logits and truth labels

        Args:
            logits: model logits
            label_ids: truth label ids
        """
        preds = self._postprocess_logits(logits)
        preds = preds.numpy()
        label_ids = label_ids.numpy()
        result = self.metric_fn(preds, label_ids)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        return result


    def update_best_model(self, dev_data_set, best_dev, best_result_file, loss, epoch, save_path=None):
        new_best_dev = best_dev
        res = self._get_eval(dev_data_set, "dev")
        dev = res[next(iter(res))] # compare first element of the evaluation metric keys
        if dev > best_dev: 
            new_best_dev = dev
            if best_result_file is not None:
                with open(best_result_file, "a+") as f:
                    f.write(
                        "best on dev= " + str(new_best_dev) + ", loss= " + str(loss) + ", epoch= " + str(epoch) + "\n"
                    )
        logger.info("Best result: Dev=%s", str(new_best_dev))
        if save_path is not None:
            self.save_model(save_path)
        return new_best_dev

    def _get_eval(self, ds, set_name):
        if ds is not None:
            preds, out_label_ids = self.evaluate(ds)
            res = self.evaluate_predictions(preds, out_label_ids)
            return res
        return None


    def inference(
        self,
        examples: List[SequenceClsInputExample],
        max_seq_length: int,
        batch_size: int = 64,
        evaluate=False,
    ):
        """
        Run inference on given examples

        Args:
            examples (List[SequenceClsInputExample]): examples
            batch_size (int, optional): batch size. Defaults to 64.

        Returns:
            logits
        """
        data_set = self.convert_to_tensors(
            examples, max_seq_length=max_seq_length, include_labels=evaluate
        )
        inf_sampler = SequentialSampler(data_set)
        inf_dataloader = DataLoader(data_set, sampler=inf_sampler, batch_size=batch_size)
        logits = self._evaluate(inf_dataloader)
        if not evaluate:
            preds = self._postprocess_logits(logits)
        else:
            logits, label_ids = logits
            preds = self._postprocess_logits(logits)
            self.evaluate_predictions(logits, label_ids)
        return preds

    def save_model(self, output_dir: str):
        """
        Save model to path

        Args:
            output_dir (str): output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.model, os.path.join(output_dir, "model.bin"))
        with io.open(output_dir + os.sep + "labels.txt", "w", encoding="utf-8") as fw:
            for l in self.labels:
                fw.write("{}\n".format(l))
        with io.open(output_dir + os.sep + "w_vocab.dat", "wb") as fw:
            pickle.dump(self.word_vocab, fw)

    @classmethod
    def load_model(cls, model_path: str):
        """
        Load a tagger model from given path

        Args:
            model_path (str): model path

            NeuralTagger: tagger model loaded from path
        """
        # Load a trained model and vocabulary from given path
        if not os.path.exists(model_path):
            raise FileNotFoundError
        with io.open(model_path + os.sep + "labels.txt") as fp:
            labels = [l.strip() for l in fp.readlines()]

        with io.open(model_path + os.sep + "w_vocab.dat", "rb") as fp:
            w_vocab = pickle.load(fp)
        # load model.bin into
        model_file_path = model_path + os.sep + "model.bin"
        if not os.path.exists(model_file_path):
            raise FileNotFoundError
        model = torch.load(model_file_path)
        new_class = cls(model, w_vocab, labels)
        crf_file_path = model_path + os.sep + "crf.bin"
        if os.path.exists(crf_file_path):
            new_class.use_crf = True
            new_class.crf = torch.load(crf_file_path)
        else:
            new_class.use_crf = False
        return new_class

    def get_logits(self, batch):
        self.model.eval()
        inputs = self.batch_mapper(batch)
        outputs = self.model(**inputs)
        return outputs[-1]


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_ids_b=None, mask=None, mask_b=None, label_id=None):
        self.input_ids = input_ids
        self.input_ids_b = input_ids_b
        self.mask = mask
        self.mask_b = mask_b
        self.label_id = label_id

