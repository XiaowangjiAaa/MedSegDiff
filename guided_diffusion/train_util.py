import copy
import functools
import os
import torch

import blobfile as bf
import torch as th
import torch.distributed as dist

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        classifier,
        diffusion,
        data,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.dataloader = dataloader
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0

        self.use_ddp = dist.is_available() and dist.is_initialized()

        if self.use_ddp:
            self.global_batch = self.batch_size * dist.get_world_size()
            self.ddp_model = th.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[dist_util.dev().index],
                output_device=dist_util.dev().index,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        else:
            self.global_batch = self.batch_size
            self.ddp_model = self.model

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = th.optim.AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params) for _ in range(len(self.ema_rate))
            ]

    # def _load_and_sync_parameters(self):
    #     resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
    #     if resume_checkpoint:
    #         print('resume model')
    #         self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
    #         logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
    #         self.model.load_part_state_dict(
    #             dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
    #         )

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            print('Loading weights only (no optimizer state)')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"Loading model from checkpoint: {resume_checkpoint}...")

            # 不用dist_util.load_state_dict，直接torch.load更灵活
            state = torch.load(resume_checkpoint, map_location=dist_util.dev())

            # 判断是完整checkpoint还是纯权重
            if isinstance(state, dict) and "model" in state:
                self.model.load_state_dict(state["model"])
            else:
                self.model.load_state_dict(state)

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=dist_util.dev())
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"optsavedmodel{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        i = 0
        data_iter = iter(self.dataloader)
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            try:
                batch, cond, name = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch, cond, name = next(data_iter)
            self.run_step(batch, cond)
            i += 1
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()


      
    def run_step(self, batch, cond):
        batch = th.cat((batch, cond), dim=1)
        cond = {}
        sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return sample

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {k: v[i : i + self.microbatch].to(dist_util.dev()) for k, v in cond.items()}
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            losses1 = compute_losses()
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses1[0]["loss"].detach())
            losses = losses1[0]
            sample = losses1[1]
            loss = (losses["loss"] * weights + losses['loss_cal'] * 10).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            self.mp_trainer.backward(loss)
        return sample

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

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            filename = f"savedmodel{(self.step+self.resume_step):06d}.pt" if not rate else f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        with bf.BlobFile(bf.join(get_blob_logdir(), f"optsavedmodel{(self.step+self.resume_step):06d}.pt"), "wb") as f:
            th.save(self.opt.state_dict(), f)

def parse_resume_step_from_filename(filename):
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    return logger.get_dir()

def find_resume_checkpoint():
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
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
