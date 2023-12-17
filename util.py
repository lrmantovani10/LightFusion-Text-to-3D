import torch.nn.functional as F
import geometry
import os
import numpy as np
import torch
import collections
import torch.distributed as dist
import summaries
import hdf5_dataio
import training
from torch.utils.data import DataLoader


def parse_intrinsics_hdf5(raw_data, trgt_sidelength=None, invert_y=False):
    intrinsics = raw_data[()]
    intrinsics = intrinsics.decode("utf-8")

    lines = intrinsics.split("\n")

    f, cx, cy, _ = map(float, lines[0].split())
    grid_barycenter = torch.Tensor(list(map(float, lines[1].split())))
    height, width = map(float, lines[3].split())

    try:
        world2cam_poses = int(lines[4])
    except ValueError:
        world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

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

    return full_intrinsic, grid_barycenter, world2cam_poses


def light_field_point_cloud(
    light_field_fn, num_samples=64**2, outlier_rejection=True
):
    dirs = torch.normal(
        torch.zeros(1, num_samples, 3), torch.ones(1, num_samples, 3)
    ).cuda()
    dirs = F.normalize(dirs, dim=-1)

    x = (torch.rand_like(dirs) - 0.5) * 2

    D = 1
    x_prim = x + D * dirs

    st = torch.zeros(1, num_samples, 2).requires_grad_(True).cuda()
    max_norm_dcdst = torch.ones_like(st) * 0
    dcdsts = []
    for i in range(5):
        d_prim = torch.normal(
            torch.zeros(1, num_samples, 3), torch.ones(1, num_samples, 3)
        ).cuda()
        d_prim = F.normalize(d_prim, dim=-1)

        a = x + st[..., :1] * d_prim
        b = x_prim + st[..., 1:] * d_prim
        v_dir = b - a
        v_mom = torch.cross(a, b, dim=-1)
        v_norm = torch.cat((v_dir, v_mom), dim=-1) / v_dir.norm(dim=-1, keepdim=True)

        with torch.enable_grad():
            c = light_field_fn(v_norm)
            dcdst = gradient(c, st)
            dcdsts.append(dcdst)
            criterion = max_norm_dcdst.norm(dim=-1, keepdim=True) < dcdst.norm(
                dim=-1, keepdim=True
            )
            max_norm_dcdst = torch.where(criterion, dcdst, max_norm_dcdst)

    dcdsts = torch.stack(dcdsts, dim=0)
    dcdt = dcdsts[..., 1:]
    dcds = dcdsts[..., :1]

    d = D * dcdt / (dcds + dcdt)
    mask = d.std(dim=0) > 1e-2
    d = d.mean(0)
    d[mask] = 0.0
    d[max_norm_dcdst.norm(dim=-1) < 1] = 0.0

    return {"depth": d, "points": x + d * dirs, "colors": c}


def gradient(y, x, grad_outputs=None, create_graph=True):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=create_graph
    )[0]
    return grad


def parse_comma_separated_integers(string):
    return list(map(int, string.split(",")))


def convert_image(img, type):
    """Expects single batch dimesion"""
    img = img.squeeze(0)

    if not "normal" in type:
        img = detach_all(lin2img(img, mode="np"))

    if "rgb" in type or "normal" in type:
        img += 1.0
        img /= 2.0
    elif type == "depth":
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img *= 255.0
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img


def flatten_first_two(tensor):
    b, s, *rest = tensor.shape
    return tensor.view(b * s, *rest)


def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, "r") as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    if trgt_sidelength is not None:
        cx = cx / width * trgt_sidelength
        cy = cy / height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array(
        [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0], [0.0, 0, 1, 0], [0, 0, 0, 1]]
    )

    return full_intrinsic, grid_barycenter, scale, world2cam_poses


def num_divisible_by_2(number):
    i = 0
    while not number % 2:
        number = number // 2
        i += 1

    return i


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d" % params)


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
    for i in range(5):
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

    dcdsts_var = torch.std(dcdsts.norm(dim=-1, keepdim=True), dim=0, keepdim=True)
    depth_var = torch.std(all_depth_estimates, dim=0, keepdim=True)

    d = D * dcdt / (dcds + dcdt)
    d[torch.abs(dcds + dcdt) < 5] = 0.0
    d[d < 0] = 0.0
    d[depth_var[0, ..., 0] > 0.01] = 0.0
    return {"depth": d, "points": x + d * plucker_coords[..., :3]}


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


def get_mgrid(sidelen, dim=2, flatten=False):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[: sidelen[0], : sidelen[1]], axis=-1)[
            None, ...
        ].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(
            np.mgrid[: sidelen[0], : sidelen[1], : sidelen[2]], axis=-1
        )[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError("Not implemented for dim=%d" % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.0
    pixel_coords = torch.from_numpy(pixel_coords)

    if flatten:
        pixel_coords = pixel_coords.view(-1, dim)
    return pixel_coords


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


# Trainer class
class Trainer:
    # Initializing parameters
    def __init__(self, model, optimizers, loss_fn, val_loss_fn, opt, rank=0):
        self.model = model
        self.opt = opt
        self.optimizers = optimizers
        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn
        self.rank = rank

    # Syncing model trained on multiple GPUs
    def sync_model(self):
        for param in self.model.parameters():
            dist.broadcast(param.data, 0)

    # Dataloader callback
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
            drop_last=True,
            num_workers=0,
        )
        return train_loader

    # Training function
    def train(self, gpu):
        if self.opt.gpus > 1:
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:1492",
                world_size=self.opt.gpus,
                rank=gpu,
            )

        # Fit the model to the device type
        self.model.to(self.opt.device)

        if self.opt.checkpoint_path is not None:
            state_dict = torch.load(self.opt.checkpoint_path)
            self.model.load_state_dict(state_dict)

        # Sync the model if multiple GPUs are used
        if self.opt.gpus > 1:
            self.sync_model()

        # Summarize outputs function
        summary_fn = summaries.img_summaries

        # Root path for logging
        root_path = os.path.join(self.opt.logging_root, self.opt.experiment_name)

        # Run the multiscale training function
        training.multiscale_training(
            model=self.model,
            dataloader_callback=self.dataloader_callback,
            dataloader_iters=(10000, 500000),
            dataloader_params=(
                (self.opt.sidelens[0], self.opt.batch_sizes[0], None),
                (self.opt.sidelens[1], self.opt.batch_sizes[1], None),
            ),
            epochs=self.opt.num_epochs,
            lr=self.opt.lr,
            steps_til_summary=self.opt.steps_til_summary,
            epochs_til_checkpoint=self.opt.epochs_til_ckpt,
            model_dir=root_path,
            loss_fn=self.loss_fn,
            val_loss_fn=self.val_loss_fn,
            iters_til_checkpoint=self.opt.iters_til_ckpt,
            summary_fn=summary_fn,
            overwrite=True,
            optimizers=self.optimizers,
            rank=gpu,
            train_function=training.train,
            gpus=self.opt.gpus,
            device=self.opt.device,
        )
