from omegaconf import DictConfig
import os
import torch
from typing import *
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from ..datasets.mimic_lab import LAST_CAREUNIT
import wandb
import pandas as pd
from sklearn import metrics
from PIL import Image
import skimage


class LightningClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.metrics = {}
        # Log config hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = self.get_optimizer()
        self.scheduler, self.scheduler_metric = self.get_scheduler()
        optimizer_config = {'optimizer': self.optimizer,
                            'lr_scheduler': self.scheduler, 'monitor': self.scheduler_metric}
        if self.scheduler is None:
            optimizer_config.pop('lr_scheduler')
            optimizer_config.pop('monitor')
        return optimizer_config

    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch, batch_idx, 'Train')

    def validation_step(self, val_batch, batch_idx):
        return self._step(val_batch, batch_idx, 'Val')

    def test_step(self, test_batch, batch_idx):
        return self._step(test_batch, batch_idx, 'Test')

    def compute_loss(self, prediction, label):
        return self.loss_fnc()(prediction, label)

    def get_prediction(self, image):
        return self(image)

    def _step(self, batch, batch_idx, prefix: str):
        ehr, img, targets_ehr, targets_cxr, seq_length, pairs, age, gender, ethnicity = batch['ehr'], batch[
            'img'], batch['targets_ehr'], batch['targets_cxr'], batch['seq_length'], batch['pairs'], batch['age'], batch['gender'], batch['ethnicity']
        self.label_names = ['mortality'] #elem[0] for elem in label_names]
        prediction = self.get_prediction((img, ehr, age, gender, ethnicity))
        loss = self.compute_loss(prediction, targets_ehr)
        prediction_prob = self._get_prediction_probability(prediction)

        self.log(f'{prefix}/Step/Loss', loss)
        return {'loss': loss, 'prediction': prediction_prob, 'label': targets_ehr} #, 'id': id, 'dataset_idx': dataset_idx}

    def _get_prediction_probability(self, prediction):
        return torch.sigmoid(prediction)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self._epoch_end(outputs, 'Train')

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self._epoch_end(outputs, 'Val')

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self._epoch_end(outputs, 'Test')

    def _epoch_end(self, outputs: List[Any], prefix: str):
        epoch_loss = torch.stack([val['loss'] for val in outputs]).mean()
        predictions_for_all_classes, labels_for_all_classes = self.get_sequential_prediction_and_labels_all_classes(
            outputs)

        metric_values = {}
        num_classes = len(predictions_for_all_classes)
        for class_idx in range(num_classes):
            metric_values.update({metric + '_' + self.label_names[class_idx]:
                                  fn(labels_for_all_classes[class_idx],
                                     predictions_for_all_classes[class_idx])
                                  for metric, fn in self.metrics.items()})

        self.log(
            f'{prefix}/Epoch/Loss', epoch_loss, on_epoch=True)

        mean_auc_over_all_classes_ = []
        for key, value in metric_values.items():
            self.log(
                f'{prefix}/Epoch/{key}', value, on_epoch=True)
            if key.startswith('AUC_'):
                mean_auc_over_all_classes_.append(value)

        mean_auc_over_all_classes = np.array(mean_auc_over_all_classes_).mean()
        self.log(
            f'{prefix}/Epoch/Mean AUC', mean_auc_over_all_classes, on_epoch=True)

        # Log for checkpoint monitoring
        self.log(f'{prefix}_epoch_loss', epoch_loss, on_epoch=True)
        self.log(f'{prefix}_mean_auc',
                 mean_auc_over_all_classes, on_epoch=True)


    def get_sequential_prediction_and_labels_all_classes(self, outputs):
        aggregated = {}
        num_classes = len(outputs[0]['prediction'][0])

        # create list of lists where each sublist contains the class-wise predictions for each sample
        # e.g. [ s1-[cA, cB, cC], s2-[cA, cB, cC], s3-[cA, cB, cC] ] such that here the first sublist
        # contains all predictions from the first sample for all classes (s1 stands for sample 1)
        # len(vals['key']) defines the batch_size of the current step
        for key in ['prediction', 'label']:
            val = [vals[key][sample_idx, :].tolist()
                   for vals in outputs for sample_idx in range(len(vals[key]))]
            aggregated[key] = val

        # create list of lists where each sublist contains the class-wise predictions for all samples
        # e.g. [ [cA_s1, cA_s2, cA_s3], [cB_s1, cB_s2, cB_s3] ]
        predictions_for_all_classes = [torch.Tensor(aggregated['prediction'])[:, class_idx].tolist()
                                       for class_idx in range(num_classes)]
        labels_for_all_classes = [torch.Tensor(aggregated['label'])[:, class_idx].tolist()
                                  for class_idx in range(num_classes)]

        return predictions_for_all_classes, labels_for_all_classes

    def get_optimizer(self):
        if self.cfg.optimizer.name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.cfg.optimizer.lr, weight_decay=0.1)
        if self.cfg.optimizer.name == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.cfg.optimizer.lr)
        return optimizer

    def get_scheduler(self):
        scheduler = None
        scheduler_metric = None
        if self.cfg.scheduler.lr_scheduler == "cosine_annealing_warm_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.cfg.scheduler.T_0, T_mult=self.cfg.scheduler.T_mult, eta_min=self.cfg.scheduler.eta_min)
        if self.cfg.scheduler.lr_scheduler == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.trainer.max_epochs, eta_min=self.cfg.scheduler.eta_min)
        if self.cfg.scheduler.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=self.cfg.scheduler.patience, factor=self.cfg.scheduler.scheduler_factor,
                threshold=1e-4, verbose=True, mode='max')
            scheduler_metric = 'Val_mean_auc'
        elif self.cfg.scheduler.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=self.cfg.scheduler.gamma)
            scheduler_metric = None

        return scheduler, scheduler_metric

    @ property
    def metrics(self):
        return self._metrics

    @ metrics.setter
    def metrics(self, metrics: dict):
        self._metrics = metrics

    @ property
    def loss_fnc(self):
        return self._loss_fnc

    @ loss_fnc.setter
    def loss_fnc(self, loss_fnc):
        self._loss_fnc = loss_fnc
