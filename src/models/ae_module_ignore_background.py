# -*- coding: utf-8 -*-
# @Time    : 2024/8/19 下午12:20
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : ae_module_ignore_background.py
# @Software: PyCharm
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from prettytable import PrettyTable
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy, Precision, F1Score
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore

from src.data.components.grass import Grass


class AELitModule(LightningModule):
    def __init__(self, net: torch.nn.Module, num_classes: int, criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, compile: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=['net'])
        self.net = net
        self.example_input_array = torch.zeros(size=(2,3,256,256))

        self.train_acc = MulticlassAccuracy(num_classes=self.hparams.num_classes, average="none", ignore_index=0)
        self.train_precision = Precision(task="multiclass", num_classes=self.hparams.num_classes, average="none",
                                         ignore_index=0)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes, average="none", ignore_index=0)
        self.train_iou = MeanIoU(num_classes=self.hparams.num_classes, include_background=False, per_class=True)
        self.train_dice = GeneralizedDiceScore(num_classes=self.hparams.num_classes, include_background=False,
                                               per_class=True)

        self.val_acc = MulticlassAccuracy(num_classes=self.hparams.num_classes, average="none", ignore_index=0)
        self.val_precision = Precision(task="multiclass", num_classes=self.hparams.num_classes, average="none",
                                       ignore_index=0)
        self.val_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes, average="none", ignore_index=0)
        self.val_iou = MeanIoU(num_classes=self.hparams.num_classes, include_background=False, per_class=True)
        self.val_dice = GeneralizedDiceScore(num_classes=self.hparams.num_classes, include_background=False,
                                             per_class=True)

        self.test_acc = MulticlassAccuracy(num_classes=self.hparams.num_classes, average="none", ignore_index=0)
        self.test_precision = Precision(task="multiclass", num_classes=self.hparams.num_classes, average="none",
                                        ignore_index=0)
        self.test_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes, average="none", ignore_index=0)
        self.test_iou = MeanIoU(num_classes=self.hparams.num_classes, include_background=False, per_class=True)
        self.test_dice = GeneralizedDiceScore(num_classes=self.hparams.num_classes, include_background=False,
                                              per_class=True)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.train_acc.reset()
        self.train_precision.reset()
        self.train_f1.reset()
        self.train_iou.reset()
        self.train_dice.reset()
        self.train_loss.reset()

        self.val_acc.reset()
        self.val_precision.reset()
        self.val_f1.reset()
        self.val_iou.reset()
        self.val_dice.reset()
        self.val_loss.reset()

        self.test_acc.reset()
        self.test_precision.reset()
        self.test_f1.reset()
        self.test_iou.reset()
        self.test_dice.reset()
        self.test_loss.reset()

    def model_step(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch['image'], batch['mask']
        logits = self.forward(x)
        loss = self.hparams.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        acc = self.train_acc(preds, targets)[1:].mean().item()
        precision = self.train_precision(preds, targets)[1:].mean().item()
        f1score = self.train_f1(preds, targets)[1:].mean().item()
        iou = self.train_iou(preds, targets).mean().item()
        dice = self.train_dice(preds, targets).mean().item()

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/macc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mprecision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mf1score", f1score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/miou", iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mdice", dice, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        all_acc = self.train_acc.compute()[1:]
        all_precision = self.train_precision.compute()[1:]
        all_f1score = self.train_f1.compute()[1:]
        all_iou = self.train_iou.compute()
        all_dice = self.train_dice.compute()

        macc = all_acc.mean()
        mprecision = all_precision.mean()
        mf1 = all_f1score.mean()
        miou = all_iou.mean()
        mdice = all_dice.mean()
        data1 = dict(macc=macc, mprecision=mprecision, mf1=mf1, miou=miou, mdice=mdice)
        data2 = dict(classes=Grass.METAINFO['classes'][1:], acc=all_acc, iou=all_iou, dice=all_dice,
                     precision=all_precision, f1score=all_f1score)
        res = self.make_table(data1, data2, stage="train")
        print(res)

        self.train_acc.reset()
        self.train_precision.reset()
        self.train_f1.reset()
        self.train_iou.reset()
        self.train_dice.reset()
        self.train_loss.reset()

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        acc = self.val_acc(preds, targets)[1:].mean().item()
        precision = self.val_precision(preds, targets)[1:].mean().item()
        f1score = self.val_f1(preds, targets)[1:].mean().item()
        iou = self.val_iou(preds, targets).mean().item()
        dice = self.val_dice(preds, targets).mean().item()

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/macc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mprecision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mf1score", f1score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/miou", iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mdice", dice, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        all_acc = self.val_acc.compute()[1:]
        all_precision = self.val_precision.compute()[1:]
        all_f1score = self.val_f1.compute()[1:]
        all_iou = self.val_iou.compute()
        all_dice = self.val_dice.compute()

        macc = all_acc.mean()
        mprecision = all_precision.mean()
        mf1 = all_f1score.mean()
        miou = all_iou.mean()
        mdice = all_dice.mean()
        data1 = dict(macc=macc, mprecision=mprecision, mf1=mf1, miou=miou, mdice=mdice)
        data2 = dict(classes=Grass.METAINFO['classes'][1:], acc=all_acc, iou=all_iou, dice=all_dice,
                     precision=all_precision, f1score=all_f1score)
        res = self.make_table(data1, data2, stage="val")
        print(res)

        self.val_acc.reset()
        self.val_precision.reset()
        self.val_f1.reset()
        self.val_iou.reset()
        self.val_dice.reset()
        self.val_loss.reset()

    def make_table(self, data1: Dict, data2: Dict, stage="train") -> str:
        table1 = PrettyTable(field_names=['meanAccuracy', 'meanPrecision', 'meanF1Score', 'meanIoU', 'meanDice'])
        table1.add_row([data1['macc'].item(), data1['mprecision'].item(), data1['mf1'].item(), data1['miou'].item(), data1['mdice'].item()])

        table2 = PrettyTable()

        for key, value in data2.items():
            if isinstance(value,tuple):
                table2.add_column(key, value)
            else:
                table2.add_column(key, value.cpu().numpy().tolist())
        return table1.get_string(title=f"{stage}:平均指标") + "\n\n" + table2.get_string(title=f"{stage}:各个类别指标")

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        acc = self.test_acc(preds, targets)[1:].mean().item()
        precision = self.test_precision(preds, targets)[1:].mean().item()
        f1score = self.test_f1(preds, targets)[1:].mean().item()
        iou = self.test_iou(preds, targets).mean().item()
        dice = self.test_dice(preds, targets).mean().item()

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/macc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mprecision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mf1score", f1score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/miou", iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mdice", dice, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        all_acc = self.test_acc.compute()[1:]
        all_precision = self.test_precision.compute()[1:]
        all_f1score = self.test_f1.compute()[1:]
        all_iou = self.test_iou.compute()
        all_dice = self.test_dice.compute()

        macc = all_acc.mean()
        mprecision = all_precision.mean()
        mf1 = all_f1score.mean()
        miou = all_iou.mean()
        mdice = all_dice.mean()
        data1 = dict(macc=macc, mprecision=mprecision, mf1=mf1, miou=miou, mdice=mdice)
        data2 = dict(classes=Grass.METAINFO['classes'][1:], acc=all_acc, iou=all_iou, dice=all_dice,
                     precision=all_precision, f1score=all_f1score)
        res = self.make_table(data1, data2, stage="test")
        print(res)

        self.test_acc.reset()
        self.test_precision.reset()
        self.test_f1.reset()
        self.test_iou.reset()
        self.test_dice.reset()
        self.test_loss.reset()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
