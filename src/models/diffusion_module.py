from typing import Any, Dict, Tuple, List
import functools
import copy
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, Precision, F1Score
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from collections import OrderedDict
from src.guided_diffusion.gaussian_diffusion import GaussianDiffusion
from src.data.components.grass import Grass
from src.guided_diffusion.resample import create_named_schedule_sampler
from src.guided_diffusion.nn import update_ema
from src.guided_diffusion.script_util import create_gaussian_diffusion


class DiffusionLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        ema_rate: List[float],
        schedule_name: str,
        drop_rate: float,
        optimizer,
        num_classes: int = 6,
        # diffusion 配置
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        b_map_scheduler_type="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
        image_size=256,
        b_map_min=1.0,
        dataset_mode="camus",
        preserve_length=False,
        add_buffer=False,
        compile=False,
        scheduler=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.diffusion = create_gaussian_diffusion(
            steps=steps,
            learn_sigma=learn_sigma,
            sigma_small=sigma_small,
            noise_schedule=noise_schedule,
            b_map_scheduler_type=b_map_scheduler_type,
            use_kl=use_kl,
            predict_xstart=predict_xstart,
            rescale_timesteps=rescale_timesteps,
            rescale_learned_sigmas=rescale_learned_sigmas,
            timestep_respacing=timestep_respacing,
            image_size=image_size,
            b_map_min=b_map_min,
            dataset_mode=dataset_mode,
            preserve_length=preserve_length,
            add_buffer=add_buffer,
        )
        self.schedule_sampler = create_named_schedule_sampler(
            schedule_name, self.diffusion
        )
        self.ema_params = [
            copy.deepcopy(list(self.net.parameters()))
            for _ in range(len(self.hparams.ema_rate))
        ]
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

        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

    def preprocess_input(self, data):
        data["label"] = data["label"].long()

        label_map = data["label"]
        bs, _, h, w = label_map.size()
        nc = self.hparams.num_classes
        input_label = torch.FloatTensor(bs, nc, h, w).zero_().to(data["label"].device)
        input_semantics = input_label.scatter_(1, label_map, 1.0).to(data["label"].device)

        if self.hparams.drop_rate > 0.0:
            mask = (
                torch.rand([input_semantics.shape[0], 1, 1, 1]) > self.drop_rate
            ).float().to(data["label"].device)
            input_semantics = input_semantics * mask

        cond = {
            key: value
            for key, value in data.items()
            if key not in ["label", "instance", "path", "label_ori"]
        }
        cond["y"] = input_semantics

        return cond

    def _update_ema(self):
        for rate, params in zip(self.hparams.ema_rate, self.ema_params):
            update_ema(params, self.net.parameters(), rate=rate)

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
        # 数据缩放到[0,1],与guided-diffusion保持一致
        x: torch.Tensor = x * 2 - 1
        cond = dict(label=y.unsqueeze(dim=1))
        cond = self.preprocess_input(cond)
        t, weights = self.schedule_sampler.sample(x.shape[0], x.device)
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.net,
            x,
            t,
            model_kwargs=cond,
        )
        losses = compute_losses()
        loss = (losses["loss"] * weights).mean()
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            loss = loss.clamp(-1e6, 1e6)
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)
        self.train_loss(loss)
        self._update_ema()

        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        self.val_loss(loss)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        self.test_loss(loss)

        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

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
