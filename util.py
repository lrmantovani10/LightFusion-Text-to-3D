from PIL import Image
import torch.nn.functional as F
import geometry
import os
import numpy as np
import torch
import collections
import h5py
import glob
import matplotlib
import util
import torchvision
import skimage
import cv2


def load_rgb_hdf5(instance_ds, key):
    rgb_ds = instance_ds["rgb"]
    img_arr = np.array(rgb_ds[key][:])
    img = resize_img(Image.fromarray(img_arr))
    img = skimage.img_as_float32(img)

    # Normalization
    img -= 0.5
    img *= 2.0

    return img


def load_pose_hdf5(instance_ds, key):
    pose_ds = instance_ds["pose"]
    extrinsics = np.array(pose_ds[key][:])
    return extrinsics.astype(np.float32).squeeze()


def parse_intrinsics_hdf5(raw_data, trgt_sidelength=None, invert_y=False):
    i_arr = np.array(raw_data)

    f, cx, cy = i_arr[:3]
    height, width = i_arr[3:]

    if trgt_sidelength is not None:
        cx = cx / width * trgt_sidelength
        cy = cy / height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    full_intrinsic = np.array(
        [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0], [0.0, 0, 1, 0], [0, 0, 0, 1]]
    )

    return full_intrinsic


def getMedianImageChannels(im):
    b, g, r = cv2.split(im)
    # Remove zeros
    b = b[b != 0]
    g = g[g != 0]
    r = r[r != 0]
    # median values
    b_median = np.median(b)
    r_median = np.median(r)
    g_median = np.median(g)
    return r_median, g_median, b_median


def resize_img(img):
    image_size = img.size
    width = image_size[0]
    height = image_size[1]

    if width != height:
        bigside = width if width > height else height
        r, g, b = [int(out) for out in getMedianImageChannels(np.array(img))]
        background = Image.new("RGB", (bigside, bigside), (r, g, b))
        offset = (
            int(round(((bigside - width) / 2), 0)),
            int(round(((bigside - height) / 2), 0)),
        )

        background.paste(img, offset)
        return background

    else:
        return img


def gradient(y, x, grad_outputs=None, create_graph=True):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=create_graph
    )[0]
    return grad


def convert_image(img):
    img = img.squeeze(0)
    img = detach_all(lin2img(img, mode="np"))
    img = img.squeeze(0)
    img += 1.0
    img /= 2.0
    img *= 255.0
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)

    return img


def flatten_first_two(tensor):
    b, s, *rest = tensor.shape
    return tensor.view(b * s, *rest)


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


def add_batch_dim_to_dict(ob):
    if isinstance(ob, collections.Mapping):
        return {k: add_batch_dim_to_dict(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(add_batch_dim_to_dict(k) for k in ob)
    elif isinstance(ob, list):
        return [add_batch_dim_to_dict(k) for k in ob]
    else:
        try:
            return ob[None, ...]
        except:
            return ob


def detach_all(tensor):
    return tensor.detach().cpu().numpy()


def lin2img(tensor, image_resolution=None, mode="torch"):
    if len(tensor.shape) == 3:
        batch_size, num_samples, channels = tensor.shape
    elif len(tensor.shape) == 2:
        num_samples, channels = tensor.shape

    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    if len(tensor.shape) == 3:
        if mode == "torch":
            tensor = tensor.permute(0, 2, 1).view(batch_size, channels, height, width)
        elif mode == "np":
            tensor = tensor.view(batch_size, height, width, channels)
    elif len(tensor.shape) == 2:
        if mode == "torch":
            tensor = tensor.permute(1, 0).view(channels, height, width)
        elif mode == "np":
            tensor = tensor.view(height, width, channels)

    return tensor


def light_field_depth_map(plucker_coords, cam2world, light_field_fn):
    x = geometry.get_ray_origin(cam2world)
    D = 1
    x_prim = x + D * plucker_coords[..., :3]

    d_prim = torch.normal(
        torch.zeros_like(plucker_coords[..., :3]),
        torch.ones_like(plucker_coords[..., :3]),
    ).to(plucker_coords.device)
    d_prim = F.normalize(d_prim, dim=-1)

    dcdsts = []
    for _ in range(5):
        st = (
            ((torch.rand_like(plucker_coords[..., :2]) - 0.5) * 1e-2)
            .requires_grad_(True)
            .to(plucker_coords.device)
        )
        a = x + st[..., :1] * d_prim
        b = x_prim + st[..., 1:] * d_prim

        v_dir = b - a
        v_mom = torch.cross(a, b, dim=-1)
        v_norm = torch.cat((v_dir, v_mom), dim=-1) / v_dir.norm(dim=-1, keepdim=True)

        with torch.enable_grad():
            c = light_field_fn(v_norm)
            dcdst = gradient(c, st, create_graph=False)
            dcdsts.append(dcdst)
            del dcdst
            del c

    dcdsts = torch.stack(dcdsts, dim=0)

    dcdt = dcdsts[0, ..., 1:]
    dcds = dcdsts[0, ..., :1]

    all_depth_estimates = D * dcdsts[..., 1:] / (dcdsts.sum(dim=-1, keepdim=True))
    all_depth_estimates[torch.abs(dcdsts.sum(dim=-1)) < 5] = 0
    all_depth_estimates[all_depth_estimates < 0] = 0.0

    depth_var = torch.std(all_depth_estimates, dim=0, keepdim=True)

    d = D * dcdt / (dcds + dcdt)
    d[torch.abs(dcds + dcdt) < 5] = 0.0
    d[d < 0] = 0.0
    d[depth_var[0, ..., 0] > 0.01] = 0.0
    return {"depth": d, "points": x + d * plucker_coords[..., :3]}


def dict_to_gpu(ob):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(dict_to_gpu(k) for k in ob)
    elif isinstance(ob, list):
        return [dict_to_gpu(k) for k in ob]
    else:
        try:
            return ob.cuda()
        except:
            return ob


def assemble_model_input(context, query, gpu=True):
    context["mask"] = torch.Tensor([1.0])
    query["mask"] = torch.Tensor([1.0])

    context = add_batch_dim_to_dict(context)
    context = add_batch_dim_to_dict(context)

    query = add_batch_dim_to_dict(query)
    query = add_batch_dim_to_dict(query)

    model_input = {"context": context, "query": query, "post_input": query}

    if gpu:
        model_input = dict_to_gpu(model_input)
    return model_input


def glob_imgs(path):
    imgs = []
    for ext in ["*.png", "*.jpg", "*.JPEG", "*.JPG"]:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def visualize_data(filepath, instance_num):
    file = h5py.File(
        filepath,
        "r",
    )

    instance = file["instance_" + str(instance_num)]
    color_keys = sorted(list(instance["rgb"].keys()))
    pose_keys = sorted(list(instance["pose"].keys()))

    img = np.array(load_rgb_hdf5(instance, color_keys[0]))
    img = skimage.img_as_ubyte(img)
    img = Image.fromarray(img)
    img.show()

    print(load_pose_hdf5(instance, pose_keys[0]))
    print(parse_intrinsics_hdf5(instance["intrinsics.txt"]))


# Testing with a custom example I made
# You might notice that the lighting is a bit darker here -- this is
# because of the normalization step in the load_rgb_hdf5 function
def test_example():
    visualize_data("image_data/cyberpunk_mercenary_with_a_tech_armor.hdf5", 1)


def image_loss(model_out, gt, mask=None):
    gt_rgb = gt["rgb"]
    return torch.nn.MSELoss()(gt_rgb, model_out["rgb"]) * 200


class LFLoss:
    def __init__(self, l2_weight=1, reg_weight=1e2):
        self.l2_weight = l2_weight
        self.reg_weight = reg_weight

    def __call__(self, model_out, gt):
        loss_dict = {}
        loss_dict["img_loss"] = image_loss(model_out, gt)
        loss_dict["reg"] = (model_out["z"] ** 2).mean() * self.reg_weight
        return loss_dict


def img_summaries(
    model_input,
    ground_truth,
    model_output,
    writer,
    iter,
    prefix="",
    img_shape=None,
):
    matplotlib.use("Agg")
    predictions = model_output["rgb"]
    trgt_imgs = ground_truth["rgb"]
    indices = model_input["query"]["instance_idx"]

    predictions = flatten_first_two(predictions)
    trgt_imgs = flatten_first_two(trgt_imgs)

    with torch.no_grad():
        if "context" in model_input and model_input["context"]:
            context_images = (
                model_input["context"]["rgb"]
                * model_input["context"]["mask"][..., None]
            )
            context_images = lin2img(
                flatten_first_two(context_images), image_resolution=img_shape
            )
            writer.add_image(
                prefix + "context_images",
                torchvision.utils.make_grid(
                    context_images, scale_each=False, normalize=True
                )
                .cpu()
                .numpy(),
                iter,
            )

        output_vs_gt = torch.cat((predictions, trgt_imgs), dim=0)
        output_vs_gt = lin2img(output_vs_gt, image_resolution=img_shape)
        writer.add_image(
            prefix + "output_vs_gt",
            torchvision.utils.make_grid(output_vs_gt, scale_each=False, normalize=True)
            .cpu()
            .detach()
            .numpy(),
            iter,
        )

        writer.add_scalar(prefix + "out_min", predictions.min(), iter)
        writer.add_scalar(prefix + "out_max", predictions.max(), iter)

        writer.add_scalar(prefix + "trgt_min", trgt_imgs.min(), iter)
        writer.add_scalar(prefix + "trgt_max", trgt_imgs.max(), iter)

        writer.add_scalar(prefix + "idx_min", indices.min(), iter)
        writer.add_scalar(prefix + "idx_max", indices.max(), iter)


def get_psnr(p, trgt):
    p = lin2img(p.squeeze(), mode="np")
    trgt = lin2img(trgt.squeeze(), mode="np")

    p = detach_all(p)
    trgt = detach_all(trgt)

    p = (p / 2.0) + 0.5
    p = np.clip(p, a_min=0.0, a_max=1.0)
    trgt = (trgt / 2.0) + 0.5

    ssim = skimage.metrics.structural_similarity(
        p, trgt, multichannel=True, data_range=1, win_size=7, channel_axis=2
    )
    psnr = skimage.metrics.structural_similarity(
        p, trgt, data_range=1, win_size=7, channel_axis=2
    )

    return psnr, ssim


def test_results(log_dir, model, dataset, save_first_n, gpu_avail):
    psnrs = []
    with torch.no_grad():
        for i in range(len(dataset)):
            print(f"Object {i:04d}")

            dummy_query = dataset[i][0]
            instance_name = dummy_query["instance_name"]

            if i < save_first_n:
                instance_dir = log_dir + f"{instance_name}"
                os.makedirs(instance_dir, exist_ok=True)

            for j, query in enumerate(dataset[i]):
                model_input = assemble_model_input(query, query, gpu_avail)
                model_output = model(model_input)

                # Obtaining the generated image and the ground truth image
                out_dict = {}
                out_dict["rgb"] = model_output["rgb"]
                out_dict["gt_rgb"] = model_input["query"]["rgb"]

                psnr, ssim = get_psnr(out_dict["rgb"], out_dict["gt_rgb"])
                psnrs.append((psnr, ssim))

                # Saving the images in the logging folder
                if i < save_first_n:
                    img = convert_image(out_dict["gt_rgb"])
                    cv2.imwrite(str(instance_dir + f"{j:06d}_gt.png"), img)
                    img = convert_image(out_dict["rgb"])
                    cv2.imwrite(str(instance_dir + f"{j:06d}.png"), img)

            print("Mean PSNRs", np.mean(np.array(psnrs), axis=0))

    with open(os.path.join(log_dir, "results.txt"), "w") as out_file:
        mean = np.mean(psnrs, axis=0)
        out_file.write(f"{mean[0]} PSRN {mean[1]} SSIM")
