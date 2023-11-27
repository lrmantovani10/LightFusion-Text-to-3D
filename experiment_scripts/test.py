# Enable import from parent package
import torch.nn.functional as F
import sys
import os
import numpy as np
import skimage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hdf5_dataio
import cv2
import torch
import models
import configargparse
import config
import util
from pathlib import Path

p = configargparse.ArgumentParser()
p.add_argument("--data_root", type=str, required=True)
p.add_argument("--logging_root", type=str, default=config.results_root)
p.add_argument("--checkpoint_path", required=True)
p.add_argument("--experiment_name", type=str, required=True)
p.add_argument("--network", type=str, default="relu")
p.add_argument("--conditioning", type=str, default="hyper")
p.add_argument("--max_num_instances", type=int, default=None)
p.add_argument(
    "--save_out_first_n",
    type=int,
    default=100,
    help="Only saves images of first n object instances.",
)
p.add_argument("--img_sidelength", type=int, default=64, required=False)

opt = p.parse_args()

state_dict = torch.load(opt.checkpoint_path)
num_instances = state_dict["latent_codes.weight"].shape[0]

model = models.LFAutoDecoder(
    num_instances=num_instances,
    latent_dim=256,
    parameterization="plucker",
    network=opt.network,
    conditioning=opt.conditioning,
)
# Checking if GPU is available
gpu_avail = torch.cuda.is_available()
if gpu_avail:
    model = model.cuda()

model.eval()
print("Loading model")
model.load_state_dict(state_dict)


def convert_image(img, type):
    img = img[0]

    if not "normal" in type:
        img = util.lin2img(img)[0]
    img = img.cpu().numpy().transpose(1, 2, 0)

    if "rgb" in type or "normal" in type:
        img += 1.0
        img /= 2.0
    elif type == "depth":
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img *= 255.0
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img


def get_psnr(p, trgt):
    p = util.lin2img(p.squeeze(), mode="np")
    trgt = util.lin2img(trgt.squeeze(), mode="np")

    p = util.detach_all(p)
    trgt = util.detach_all(trgt)

    p = (p / 2.0) + 0.5
    p = np.clip(p, a_min=0.0, a_max=1.0)
    trgt = (trgt / 2.0) + 0.5

    ssim = skimage.metrics.structural_similarity(
        p, trgt, multichannel=True, data_range=1, win_size=7, channel_axis=2
    )
    psnr = skimage.metrics.structural_similarity(p, trgt, data_range=1, win_size=7, channel_axis=2)

    return psnr, ssim


print("Loading dataset")
dataset = hdf5_dataio.get_instance_datasets_hdf5(
    opt.data_root, sidelen=opt.img_sidelength, max_num_instances=opt.max_num_instances
)
log_dir = Path(opt.logging_root) / opt.experiment_name

psnrs = []

with torch.no_grad():
    for i in range(len(dataset)):
        print(f"Object {i:04d}")

        dummy_query = dataset[i][0]
        instance_name = dummy_query["instance_name"]

        if i < opt.save_out_first_n:
            instance_dir = log_dir / f"{instance_name}"
            instance_dir.mkdir(exist_ok=True, parents=True)

        for j, query in enumerate(dataset[i]):
            model_input = util.assemble_model_input(query, query, gpu_avail)
            model_output = model(model_input)

            # Obtaining the generated image and the ground truth image
            out_dict = {}
            out_dict["rgb"] = model_output["rgb"]
            out_dict["gt_rgb"] = model_input["query"]["rgb"]

            is_context = False
            psnr, ssim = get_psnr(out_dict["rgb"], out_dict["gt_rgb"])
            psnrs.append((psnr, ssim))

            # Saving the images on the logging folder
            if i < opt.save_out_first_n:
                img = convert_image(out_dict["gt_rgb"], "rgb")
                cv2.imwrite(str(instance_dir / f"{j:06d}_gt.png"), img)
                img = convert_image(out_dict["rgb"], "rgb")
                cv2.imwrite(str(instance_dir / f"{j:06d}.png"), img)

        print(np.mean(np.array(psnrs), axis=0))

with open(os.path.join(log_dir, "results.txt"), "w") as out_file:
    mean = np.mean(psnrs, axis=0)
    out_file.write(f"{mean[0]} PSRN {mean[1]} SSIM")
