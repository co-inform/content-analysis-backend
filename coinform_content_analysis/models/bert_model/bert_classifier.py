# -*- coding: utf-8 -*-
import logging as log
import sys
from collections import OrderedDict
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
from test_tube import HyperOptArgumentParser
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
from transformers import BertModel, get_constant_schedule_with_warmup

from bert_tokenizer import BERTTextEncoder
from coinform_content_analysis.data_loader.data_utils import rumoureval_veracity_dataset, rumoureval_stance_dataset
from utils import mask_fill
from sklearn.metrics import f1_score

class BERTClassifier(pl.LightningModule):
    """
    Sample model to show how to use BERT to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams) -> None:
        super(BERTClassifier, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if hparams.nr_frozen_epochs > 0:
            self._frozen = True
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )

        # Tokenizer
        self.tokenizer = BERTTextEncoder("bert-base-uncased")

        self.__label_encoder()

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(self.hparams.dropout),
            nn.Linear(768, self.label_encoder.vocab_size),
        )

    def __label_encoder(self):
        task = self.hparams.task

        ## rumoureval stance detection label encoder
        if task == 'subtaskaenglish':
            self.label_set = {
                "comment": 0,
                "query": 1,
                "deny": 2,
                "support": 3
            }
        ## rumoureval veracity detection label encoder
        else:
            self.label_set = {
                "false": 0,
                "uverified": 1,
                "true": 2
            }
        self.label_encoder = LabelEncoder(
            list(self.label_set.keys()), reserved_labels=[]
        )
        log.info('Num of class {}'.format(self.label_encoder.vocab_size))

    def __build_loss(self):
        """ Initializes the loss function/s. """
        if self.hparams.class_weights != "ignore":
            weights = [float(x) for x in self.hparams.class_weights.split(",")]
            self._loss = nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32), reduction="sum"
            )
        else:
            self._loss = nn.CrossEntropyLoss(reduction="sum")

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.bert.parameters():
                param.requires_grad = True

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, tokens, lengths):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        tokens = tokens[:, : lengths.max()]
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        mask = lengths_to_mask(lengths, device=tokens.device)

        # Run BERT model.
        word_embeddings, _, _ = self.bert(tokens, mask)

        # Average Pooling
        word_embeddings = mask_fill(
            0.0, tokens, word_embeddings, self.tokenizer.padding_index
        )
        sentemb = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()).float().sum(1)
        sentemb = sentemb / sum_mask

        return {"logits": self.classification_head(sentemb)}

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        tokens, lengths = self.tokenizer.batch_encode(sample["text"])

        inputs = {"tokens": tokens, "lengths": lengths}

        if not prepare_target:
            return inputs, {}

        # Prepare target:
        targets = {"labels": self.label_encoder.batch_encode(sample["label"])}
        return inputs, targets

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        f1 = f1_score(y.cpu().numpy(), labels_hat.cpu().numpy(), average="macro")
        f1 = torch.from_numpy(np.asarray(f1))

        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc, "f1_macro": f1,})

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        f1_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

            # reduce manually when using dp
            f1 = output["f1_macro"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                f1_acc = torch.mean(f1)

            f1_mean += f1_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        f1_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
            "f1_mean": f1_mean
        }
        return result

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.bert.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        scheduler = get_constant_schedule_with_warmup(
            optimizer, self.hparams.warmup_steps
        )
        return [optimizer], []


    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    def __retrieve_dataset(self, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        task = self.hparams.task
        if task == 'subtaskaenglish':
            return rumoureval_stance_dataset(self.hparams, train, val, test)
        else:
            return rumoureval_veracity_dataset(self.hparams, train, val, test)

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)[0]
        log.info('Train {}'.format(len(self._train_dataset.rows)))
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @pl.data_loader
    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._dev_dataset = self.__retrieve_dataset(train=False, test=False)[0]
        log.info('Dev {}'.format(len(self._dev_dataset.rows)))
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @pl.data_loader
    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)[0]
        log.info('Test {}'.format(len(self._test_dataset.rows)))
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @classmethod
    def add_model_specific_args(
        cls, parser: HyperOptArgumentParser
    ) -> HyperOptArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: HyperOptArgumentParser obj

        Returns:
            - updated parser
        """
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--class_weights",
            default="ignore",
            type=str,
            help="Weights for each of the classes we want to tag.",
        )
        parser.opt_list(
            "--nr_frozen_epochs",
            default=sys.maxsize,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
            tunable=True,
            options=[0, 1, 2, 3, 4, 5],
        )
        parser.add_argument(
            "--warmup_steps", default=200, type=int, help="Scheduler warmup steps.",
        )
        parser.opt_list(
            "--dropout",
            default=0.1,
            type=float,
            help="Dropout to be applied to the BERT embeddings.",
            tunable=True,
            options=[0.1, 0.2, 0.3, 0.4, 0.5],
        )

        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        parser.add_argument(
            "--task",
            default='subtaskaenglish',
            help="Task type for the classification, e.g subtaskaenglish",
        )
        return parser
