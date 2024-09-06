# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 下午20:50
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : grass_datamodule.py
# @Software: PyCharm
from typing import Any, Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.grass import Grass


class GRASSDataModule(LightningDataModule):
    def __init__(
            self,
            root: str,
            train_pipeline: Dict = None,
            val_pipeline: Dict = None,
            test_pipeline: Dict = None,
            train_ratio: float = 0.8,
            batch_size: int = 1,
            num_workers: int = 0,
            pin_memory: bool = False,
            persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['train_pipeline', 'val_pipeline', 'test_pipeline'])
        self.train_pipeline = train_pipeline
        self.val_pipeline = val_pipeline
        self.test_pipeline = test_pipeline
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return len(Grass.METAINFO["classes"])

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        Grass(
            root=self.hparams.root,
            **self.train_pipeline,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            dataset = Grass(
                root=self.hparams.root,
                **self.train_pipeline,
            )
            total_nums = len(dataset)
            train_size = int(total_nums * self.hparams.train_ratio)
            val_size = total_nums - train_size
            self.train_dataset = Grass(
                root=self.hparams.root,
                phase="train"
                **self.train_pipeline,
            )
            self.val_dataset = Grass(
                root=self.hparams.root,
                phase="val"
                **self.val_pipeline,
            )
            self.test_dataset = self.val_dataset

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = GRASSDataModule()
