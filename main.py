import os
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
opt.num_images = 3
opt.lr = 1e-4
opt.num_epochs = 100
opt.steps_til_summary = 50
opt.epochs_til_ckpt = 10
opt.iters_til_ckpt = 10000
opt.logging_root = "lfn_logs/"
opt.results_root = "lfn_results/"
opt.image_folder = "image_data/"
opt.num_trgt_samples = 1
opt.max_num_instances = None
# If you have a trained model, set this to the path of the file containing the model's weights
opt.checkpoint_path = None
opt.batch_size = 50
# Side length of the squared images used by the model
opt.sidelen = 128
opt.batches_per_validation = 10
# Whether to drop the last batch
opt.drop_last = False
opt.dataloader_iters = 500000
opt.save_first_n = 100

if __name__ == "__main__":
    opt.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # For now, ignoring MPS because the PyTorch MPS implementation it is still in development -- this next condition can be removed if the MPS implementation is stable
    if opt.device == "mps":
        opt.device = "cpu"

    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    opt.model_dir = os.path.join(
        root_path, "train" if opt.train == "true" else "generated"
    )
    test_checkpoint_path = os.path.join(
        root_path, "generated/checkpoints/model_final.pth"
    )
    train_checkpoint_path = os.path.join(root_path, "train/checkpoints/model_final.pth")
    if os.path.exists(test_checkpoint_path):
        opt.checkpoint_path = test_checkpoint_path
    elif os.path.exists(train_checkpoint_path):
        opt.checkpoint_path = train_checkpoint_path
    print("PATHS", train_checkpoint_path, test_checkpoint_path, opt.checkpoint_path)
    if opt.train == "true":
        opt.data_root = generate_images(
            opt.prompt,
            style=opt.prompt_style,
            device=opt.device,
            initial_negative_prompt=opt.negative_prompt,
            image_folder=opt.image_folder,
            num_images=opt.num_images,
            final_width=opt.sidelen,
        )
    else:
        if not opt.checkpoint_path:
            raise FileNotFoundError(
                "No trained model found for the prompt: "
                + opt.experiment_name
                + ". Please train the model before attempting to generate a 3D reconstruction."
            )
        cleaned_prompt = opt.prompt.lower().replace(" ", "_").replace(".", ",")
        opt.data_root = opt.image_folder + cleaned_prompt + ".hdf5"

    num_instances = hdf5_dataio.get_num_instances(opt.data_root)
    model = models.LFAutoDecoder(
        latent_dim=256,
        num_instances=num_instances,
        parameterization="plucker",
        network=opt.network,
        conditioning=opt.conditioning,
    )

    loss_fn = val_loss_fn = util.LFLoss(reg_weight=1)
    manager = Manager()
    shared_dict = manager.dict()
    opt.cache = shared_dict

    if opt.train == "true":
        if opt.checkpoint_path:
            state_dict = torch.load(opt.checkpoint_path)
            model.load_state_dict(state_dict)
        optimizers = [torch.optim.Adam(lr=opt.lr, params=model.parameters())]

        trainer = Trainer(model, optimizers, loss_fn, val_loss_fn, opt, rank=0)
        if opt.gpus > 1:
            mp.spawn(trainer.train, nprocs=opt.gpus, join=True)
        else:
            trainer.train(0)

    else:
        # Generation Stage
        state_dict = torch.load(opt.checkpoint_path)
        state_dict["latent_codes.weight"] = torch.zeros_like(model.latent_codes.weight)
        model.load_state_dict(state_dict)
        latent_params = [
            (name, param)
            for name, param in model.named_parameters()
            if "latent_codes" in name
        ]
        optimizers = [torch.optim.Adam(lr=opt.lr, params=[p for _, p in latent_params])]
        trainer = Trainer(model, optimizers, loss_fn, val_loss_fn, opt, rank=0)

        ### TESTING

        # if opt.gpus > 1:
        #     mp.spawn(trainer.train, nprocs=opt.gpus, join=True)
        # else:
        #     trainer.train(0)

        ###

        # Testing / evaluation Stage
        trainer.model.eval()

        print("Loading dataset")

        ### TESTING
        # opt.data_root = opt.data_root.split(".")[0] + "_generated.hdf5"
        opt.data_root = opt.data_root.split(".")[0] + ".hdf5"
        ###

        dataset = hdf5_dataio.get_instance_datasets_hdf5(
            opt.data_root,
            sidelen=opt.sidelen,
            max_num_instances=opt.max_num_instances,
        )
        results_dir = os.path.join(opt.model_dir, opt.results_root)

        util.test_results(
            results_dir, trainer.model, dataset, opt.save_first_n, opt.device != "cpu"
        )
