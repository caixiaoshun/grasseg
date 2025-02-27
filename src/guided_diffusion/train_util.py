# this script has been modified to incorporate w&b 
# 07/11/2023
from typing import List, Dict

import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from src.guided_diffusion import dist_util, logger
from src.guided_diffusion.fp16_util import MixedPrecisionTrainer
from src.guided_diffusion.nn import update_ema
from src.guided_diffusion.resample import LossAwareSampler, UniformSampler

import gc
import torch
import matplotlib.pyplot as plt

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            num_classes,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            drop_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth = 1e-3,
            schedule_sampler=None,
            weight_decay=1e-3,
            lr_anneal_steps=0,
            output_dir, 
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size

        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.drop_rate = drop_rate
        self.log_interval = log_interval 
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.output_dir = output_dir 

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        
        # Clear CUDA cache before starting the training
        torch.cuda.empty_cache()


    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                print("loading model from checkpoint: ", resume_checkpoint)
                self.model.load_state_dict(
                    th.load(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

            if self.opt.param_groups[0]['lr'] != self.lr:
                self.opt.param_groups[0]['lr'] = self.lr

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            logger.log(f"step: {self.step + self.resume_step}")
            batch, cond = next(self.data)

            cond = self.preprocess_input(cond)

            self.run_step(batch, cond)
            kvs = logger.getkvs()

            if self.step % self.log_interval == 0:
                print(self.step, " steps completed, let's check the logger")
                logger.dumpkvs()

            if self.step % self.save_interval == 0 and self.step > 0:
                print("Saving step")
                logger.log("Saving step")
                self.save()
                self.sanity_test(batch=batch, device=dist_util.dev(), cond=cond)
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()
   
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)

        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

        # Clear CUDA cache after each step
        torch.cuda.empty_cache()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]

            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp: 
                losses = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            loss = (losses["loss"] * weights).mean()
            if th.isnan(loss).any() or th.isinf(loss).any():
                loss = loss.clamp(-1e6, 1e6)

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self): 
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("lr", self.opt.param_groups[0]['lr'])
        logger.logkv("lr_anneal_steps", self.lr_anneal_steps)
        logger.logkv("memory_usage", torch.cuda.memory_allocated())

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                save_path = os.path.join(self.output_dir, filename)
                th.save(state_dict, save_path)
                logger.log(f"saved model {rate} to {save_path}")

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            optimizer_filename = f"opt{(self.step + self.resume_step):06d}.pt"
            optimizer_path = os.path.join(self.output_dir, optimizer_filename)
            th.save(self.opt.state_dict(), optimizer_path)
        dist.barrier()

    def sanity_test(self, batch, device, cond):
        src_img = ((batch + 1.0) / 2.0).to(device)
        model_kwargs = cond

        with th.no_grad():
            self.model.eval()
            inference_img, snapshots = self.diffusion.p_sample_loop_with_snapshot(
                self.model,
                (batch.shape[0], 3, batch.shape[2], batch.shape[3]),
                model_kwargs=model_kwargs,
                progress=True
            )
            self.model.train()

        inference_img = (inference_img + 1) / 2.0
        log_images(inference_img=inference_img, src_img=src_img, snapshots=snapshots, output_dir=self.output_dir, self=self)

    def preprocess_input(self, data):
        data['label'] = data['label'].long()

        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.num_classes
        input_label = th.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        if 'instance' in data:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

        if self.drop_rate > 0.0:
            mask = (th.rand([input_semantics.shape[0], 1, 1, 1]) > self.drop_rate).float()
            input_semantics = input_semantics * mask

        cond = {key: value for key, value in data.items() if key not in ['label', 'instance', 'path', 'label_ori']}
        cond['y'] = input_semantics

        return cond

    def get_edges(self, t):
        edge = th.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir(self):
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return self.output_dir #logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{step :06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

            
def log_images(inference_img, src_img, snapshots, output_dir, self):
    num_rows = 2 + len(snapshots)
    num_cols = inference_img.shape[0]
    base_width = 4
    base_height = 4
    fig_width = num_cols * base_width + 2
    fig_height = num_rows * base_height

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    fig.suptitle('Diffusion Model Results', fontsize=16)

    for k in range(num_cols):
        axs[0, k].imshow(src_img[k, 0, ...].cpu().detach().numpy(), cmap='gray')
        axs[0, k].axis('off')

        axs[1, k].imshow(inference_img[k, 0, ...].cpu().detach().numpy(), cmap='gray')
        axs[1, k].axis('off')

        for i, snap in enumerate(snapshots):
            axs[i+2, k].imshow(snapshots[snap][k, 0, ...].cpu().detach().numpy(), cmap='gray')
            axs[i+2, k].axis('off')

    axs[0, 0].set_title("Source Image")
    axs[1, 0].set_title("Inference Image")
    for i, snap in enumerate(snapshots):
        axs[i+2, 0].set_title(f"Snapshot {snap}")

    plt.tight_layout(rect=[0, 0.0, 1, 0.95])  
    plt.savefig(f"{output_dir}/diffusion_results_{str(self.step).zfill(6)}.png")
    plt.close()