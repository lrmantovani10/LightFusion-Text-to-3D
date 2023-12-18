import numpy as np
from glob import glob
import os
import h5py
from PIL import Image


def load_pose_hdf5(instance_ds):
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


def glob_imgs(path):
    imgs = []
    for ext in ["*.png", "*.jpg", "*.JPEG", "*.JPG"]:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


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


# Testing data visualization with a simple "dragon.hdf5" example
def test_examples():
    visualize_data("data/dragon.hdf5")
