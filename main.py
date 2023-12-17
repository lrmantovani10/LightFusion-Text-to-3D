import configargparse
import torch
import sd
from multiprocessing import Manager
import torch.multiprocessing as mp
import hdf5_dataio
import models
import os
from sd import generate_images
import loss_functions
from util import Trainer
import promptStyles

p = configargparse.ArgumentParser()
p.add_argument("--prompt", type=str, required=True)
p.add_argument("--negative_prompt", type=str, help="What the image shouldn't be")
p.add_argument(
    "--prompt_style", type=str, choices=list(promptStyles.styles.keys()), default=None
)
p.add_argument("--train", type=str, choices=["true", "false"], default="true")

# Defining additional necessary parameters
opt = p.parse_args()
opt.gpus = torch.cuda.device_count()
opt.network = "relu"
opt.conditioning = "hyper"
opt.experiment_name = opt.prompt.replace(" ", "_").replace(".", ",")
opt.lr = 1e-4
opt.num_epochs = 100
opt.steps_til_summary = 1000
opt.epochs_til_ckpt = 10
opt.iters_til_ckpt = 10000
opt.logging_root = "lfn_logs"
opt.num_trgt_samples = 1
opt.max_num_instances = None

# Training case
if opt.train == "true":
    # Define the device
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the different poses using Stable Diffusion
    opt.data_root = generate_images(
        opt.prompt,
        style=opt.prompt_style,
        device=opt.device,
        initial_negative_prompt=opt.negative_prompt,
    )

    ### TEST IMAGE GENERATION BEFORE PROCEEDING ###

    # # Define the model
    # num_instances = hdf5_dataio.get_num_instances(opt.data_root)
    # model = models.LFAutoDecoder(
    #     latent_dim=256,
    #     num_instances=num_instances,
    #     parameterization="plucker",
    #     network=opt.network,
    #     conditioning=opt.conditioning,
    # ).to(opt.device)

    # # Define the loss
    # loss_fn = val_loss_fn = loss_functions.LFLoss(reg_weight=1)

    # # Define the Adam optimizer
    # optimizers = [torch.optim.Adam(lr=opt.lr, params=model.parameters())]

    # # Create a trainer instance
    # trainer = Trainer(model, optimizers, loss_fn, val_loss_fn, opt, rank=0)

    # # Define the manager and shared dictionary for parallel training
    # manager = Manager()
    # shared_dict = manager.dict()
    # opt.cache = shared_dict

    # # Define the batch sizes and side lengths
    # opt.batch_sizes = 256, 50
    # opt.sidelens = 64, 128

    # # Start training
    # mp.spawn(trainer.train, nprocs=opt.gpus)

# Recreation / Testing phase
else:
    pass
