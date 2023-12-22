import os
import hdf5_dataio
import shutil
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import util


class Trainer:
    def __init__(self, model, optimizers, loss_fn, val_loss_fn, opt, rank=0):
        self.model = model
        self.opt = opt
        self.optimizers = optimizers
        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn
        self.rank = rank

    def sync_model(self):
        for param in self.model.parameters():
            dist.broadcast(param.data, 0)

    def average_gradients(self):
        """Averages gradients across workers"""
        size = float(dist.get_world_size())

        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size

    def dataloader_callback(self, sidelength, batch_size, query_sparsity):
        train_dataset = hdf5_dataio.SceneClassDataset(
            num_context=0,
            num_trgt_samples=self.opt.num_trgt_samples,
            data_root=self.opt.data_root,
            query_sparsity=query_sparsity,
            img_sidelength=sidelength,
            vary_context_number=True,
            cache=self.opt.cache,
            max_num_instances=self.opt.max_num_instances,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=self.opt.drop_last,
            num_workers=0,
        )
        return train_loader

    def train_step(self, rank, overwrite=True):
        dataloaders = self.dataloader_callback(*self.dataloader_params)

        if isinstance(dataloaders, tuple):
            train_dataloader, val_dataloader = dataloaders
            assert (
                self.val_loss_fn is not None
            ), "If validation set is passed, have to pass a validation loss_fn!"
        else:
            train_dataloader, val_dataloader = dataloaders, None

        if rank == 0:
            if os.path.exists(self.opt.model_dir):
                if overwrite:
                    shutil.rmtree(self.opt.model_dir)
                else:
                    val = input(
                        "The model directory %s exists. Overwrite? (y/n)"
                        % self.opt.model_dir
                    )
                    if val == "y" or overwrite:
                        shutil.rmtree(self.opt.model_dir)

            os.makedirs(self.opt.model_dir)

            summaries_dir = os.path.join(self.opt.model_dir, "summaries")
            util.cond_mkdir(summaries_dir)

            checkpoints_dir = os.path.join(self.opt.model_dir, "checkpoints")
            util.cond_mkdir(checkpoints_dir)

            writer = SummaryWriter(summaries_dir, flush_secs=10)

        total_steps = 0
        with tqdm(total=len(train_dataloader) * self.opt.num_epochs) as pbar:
            for epoch in range(self.opt.num_epochs):
                if not epoch % self.opt.epochs_til_ckpt and epoch and rank == 0:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            checkpoints_dir,
                            "model_epoch_%04d_iter_%06d.pth" % (epoch, total_steps),
                        ),
                    )
                for model_input, gt in train_dataloader:
                    if self.opt.device != "cpu":
                        model_input = util.dict_to_gpu(model_input)
                        gt = util.dict_to_gpu(gt)

                    model_output = self.model(model_input)
                    losses = self.loss_fn(model_output, gt)

                    train_loss = 0.0
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if rank == 0:
                            writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    if rank == 0:
                        writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not total_steps % self.opt.steps_til_summary and rank == 0:
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(checkpoints_dir, "model_current.pth"),
                        )
                        for i, optim in enumerate(self.optimizers):
                            torch.save(
                                optim.state_dict(),
                                os.path.join(checkpoints_dir, f"optim_{i}_current.pth"),
                            )
                        if self.summary_fn is not None:
                            self.summary_fn(
                                model_input,
                                gt,
                                model_output,
                                writer,
                                total_steps,
                                "train_",
                            )

                    for optim in self.optimizers:
                        optim.zero_grad()
                    train_loss.backward()

                    if self.opt.gpus > 1:
                        self.average_gradients()

                    for optim in self.optimizers:
                        optim.step()
                    del train_loss

                    if rank == 0:
                        pbar.update(1)

                    if not total_steps % self.opt.steps_til_summary and rank == 0:
                        print(
                            ", ".join(
                                [f"Epoch {epoch}"]
                                + [
                                    f"{name} {loss.mean()}"
                                    for name, loss in losses.items()
                                ]
                            )
                        )

                        if val_dataloader is not None:
                            print("Running validation set...")
                            with torch.no_grad():
                                self.model.eval()
                                val_losses = defaultdict(list)
                                for val_i, (model_input, gt) in enumerate(
                                    val_dataloader
                                ):
                                    if self.opt.device != "cpu":
                                        model_input = util.dict_to_gpu(model_input)
                                        gt = util.dict_to_gpu(gt)

                                    model_output = self.model(model_input, val=True)
                                    val_loss, val_loss_smry = self.val_loss_fn(
                                        model_output, gt, val=True, model=self.model
                                    )

                                    for name, value in val_loss.items():
                                        val_losses[name].append(value)

                                    if val_i == self.opt.batches_per_validation:
                                        break

                                for loss_name, loss in val_losses.items():
                                    single_loss = np.mean(
                                        np.concatenate(
                                            [l.reshape(-1).cpu().numpy() for l in loss],
                                            axis=0,
                                        )
                                    )

                                    if rank == 0:
                                        writer.add_scalar(
                                            "val_" + loss_name,
                                            single_loss,
                                            total_steps,
                                        )

                            self.model.train()

                    if (
                        (self.opt.iters_til_ckpt is not None)
                        and (not total_steps % self.opt.iters_til_ckpt)
                        and rank == 0
                    ):
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(
                                checkpoints_dir,
                                "model_epoch_%04d_iter_%06d.pth" % (epoch, total_steps),
                            ),
                        )

                    total_steps += 1
                    if (
                        self.opt.dataloader_iters is not None
                        and total_steps == self.opt.dataloader_iters
                    ):
                        break

                if (
                    self.opt.dataloader_iters is not None
                    and total_steps == self.opt.dataloader_iters
                ):
                    break

            if rank == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(checkpoints_dir, "model_final.pth"),
                )

    def train(self, gpu=0):
        if self.opt.gpus > 1:
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:1492",
                world_size=self.opt.gpus,
                rank=gpu,
            )

        self.model.to(self.opt.device)
        if self.opt.checkpoint_path is not None:
            state_dict = torch.load(self.opt.checkpoint_path)
            self.model.load_state_dict(state_dict)

        if self.opt.gpus > 1:
            self.sync_model()

        self.summary_fn = util.img_summaries
        self.dataloader_params = (self.opt.sidelen, self.opt.batch_size, None)

        self.train_step(gpu)
