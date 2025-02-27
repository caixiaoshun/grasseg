# -*- coding: utf-8 -*-
# @Time    : 2024/8/19 下午12:20
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : ae_module_allclasses.py
# @Software: PyCharm
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, Precision, F1Score
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from collections import OrderedDict

from src.data.components.grass import Grass


class AELitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        average: str = "macro",
        compile: bool = False,
        experiment_name="experiment"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.example_input_array = torch.zeros(size=(2, 3, 256, 256))

        self.train_metrics = self.get_metrics(
            prefix="train", task="multiclass", averge=self.hparams.average
        )
        self.val_metrics = self.get_metrics(
            prefix="val", task="multiclass", averge=self.hparams.average
        )
        self.test_metrics = self.get_metrics(
            prefix="test", task="multiclass", averge=self.hparams.average
        )

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def get_metrics(self, prefix: str = "train", averge="macro", task="multiclass"):
        return MetricCollection(
            {
                f"{prefix}/acc": MulticlassAccuracy(
                    num_classes=self.hparams.num_classes, average=averge
                ),
                f"{prefix}/precision": Precision(
                    task=task, num_classes=self.hparams.num_classes, average=averge
                ),
                f"{prefix}/f1Score": F1Score(
                    task=task, num_classes=self.hparams.num_classes, average=averge
                ),
                f"{prefix}/iou": MeanIoU(num_classes=self.hparams.num_classes),
                f"{prefix}/dice": GeneralizedDiceScore(
                    num_classes=self.hparams.num_classes
                ),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.val_metrics.reset()

        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

    def model_step(
        self, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch["image"], batch["mask"]
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
        self.train_metrics(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_metrics(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_metrics(preds, targets)

        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

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
