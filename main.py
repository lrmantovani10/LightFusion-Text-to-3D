import configargparse
import torch
from multiprocessing import Manager
import torch.multiprocessing as mp
import hdf5_dataio
import models
import util
from sd import generate_images
from training import Trainer
import promptStyles

p = configargparse.ArgumentParser()
p.add_argument("--prompt", type=str, required=True)
p.add_argument("--negative_prompt", type=str, help="What the image shouldn't be")
p.add_argument(
    "--prompt_style", type=str, choices=list(promptStyles.styles.keys()), default=None
)
p.add_argument("--train", type=str, choices=["true", "false"], default="true")

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
opt.logging_root = ".lfn_logs/"
opt.results_root = ".lfn_results/"
opt.num_trgt_samples = 1
opt.max_num_instances = None
# If you have a trained model, set this to the path of the file containing the model's weights
opt.checkpoint_path = None
opt.batch_sizes = 256, 50
opt.sidelens = 64, 128
opt.batches_per_validation = 10

if opt.train == "true":
    opt.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    opt.data_root = generate_images(
        opt.prompt,
        style=opt.prompt_style,
        device=opt.device,
        initial_negative_prompt=opt.negative_prompt,
    )

    num_instances = hdf5_dataio.get_num_instances(opt.data_root)
    model = models.LFAutoDecoder(
        latent_dim=256,
        num_instances=num_instances,
        parameterization="plucker",
        network=opt.network,
        conditioning=opt.conditioning,
    ).to(opt.device)

    loss_fn = val_loss_fn = util.LFLoss(reg_weight=1)
    optimizers = [torch.optim.Adam(lr=opt.lr, params=model.parameters())]
    trainer = Trainer(model, optimizers, loss_fn, val_loss_fn, opt, rank=0)

    manager = Manager()
    shared_dict = manager.dict()
    opt.cache = shared_dict

    mp.spawn(trainer.train, nprocs=opt.gpus)

# Recreation / Testing phase
else:
    pass
