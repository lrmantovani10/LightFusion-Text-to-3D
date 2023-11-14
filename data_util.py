import functools
import cv2
import numpy as np
import imageio
from glob import glob
import os
import shutil
import skimage
import h5py
import io
from PIL import Image


def load_depth(path, sidelength=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

    img *= 1e-4

    if len(img.shape) == 3:
        img = img[:, :, :1]
        img = img.transpose(2, 0, 1)
    else:
        img = img[None, :, :]
    return img


def load_rgb_hdf5(instance_ds, key, sidelength=None):
    rgb_ds = instance_ds["rgb"][()]
    # Cast to float
    rgb_ds = rgb_ds.astype(np.float64)
    img = square_crop_img(rgb_ds)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_AREA)

    img -= 0.5
    img *= 2.0

    return img


def load_pose_hdf5(instance_ds, key):
    pose_ds = instance_ds["pose"][()]
    pose = pose_ds.decode("utf-8")

    lines = pose.splitlines()

    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        # processed_pose = pose.squeeze()
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[
        center_coord[0] - min_dim // 2 : center_coord[0] + min_dim // 2,
        center_coord[1] - min_dim // 2 : center_coord[1] + min_dim // 2,
    ]
    return img


def glob_imgs(path):
    imgs = []
    for ext in ["*.png", "*.jpg", "*.JPEG", "*.JPG"]:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

'''
For visualizing the data in the original hdf5 file used by the LFN's creators:

def visualize_data(filepath):
    with h5py.File(filepath, "r") as file:
        # Setting the dataset to the first entry in the file
        for name, item in file.items():
            print(name, item)
            dataset = item
            break

        # Viewing the first pose
        pose = dataset["pose"]
        print("Pose", pose)
        for _, item in pose.items():
            data = item[()]
            text_content = data.tobytes().decode("utf-8")
            print(text_content, "\n")
            break

        # Displaying the first image
        rgb = dataset["rgb"]
        print("RGB", rgb)

        for item in rgb.keys():
            raw = dataset["rgb"][item][...]
            s = raw.tostring()
            f = io.BytesIO(s)
            # Display the image
            img = Image.open(f)
            img.show()
            break

        # Intrinsics data
        intrinsics = ["intrinsics.txt"]
        # Print the data of the first intrinsics file
        for item in intrinsics:
            intrinsics = dataset[item][...]
            # convert to string
            s = intrinsics.tostring()
            s = s.decode("utf-8")
            print(s)
            break
'''



def visualize_data(filepath):
    with h5py.File(filepath, "r") as file:
        # Setting the dataset to the first entry in the file
        for name, item in file.items():
            print(name, item)
            dataset = item
            break

        # Viewing the first pose
        pose = dataset["pose"][()]
        pose = pose.decode("utf-8")
        print("Pose\n", pose)

        # Displaying the first image
        rgb = dataset["rgb"][()]

        # Display the image
        img = Image.fromarray(rgb)
        img.show()

        # Display intrinsics data
        intrinsics = dataset["intrinsics.txt"]
        intrinsics = intrinsics[()]
        intrinsics = intrinsics.decode("utf-8")
        print("Intrinsics\n", intrinsics)


# Testing data visualization
def test_examples():
    from data_util import visualize_data
    from sd import generate_image

    generate_image("dragon", 1.5, 20, 5, 128, 128, 1)
    visualize_data("data/dragon.hdf5")
