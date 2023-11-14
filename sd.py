import torch, os, promptStyles, tomesd
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
import h5py

model_name = "stabilityai/stable-diffusion-2-1"

# Fast mode - slightly lower quality tradeoff
fast_mode = False

# Check if CUDA is available
cuda_there = False
if torch.cuda.is_available():
    import deepspeed

    cuda_there = True

# Check if MPS is available
mps_there = False
if os.environ.get("MPS", None) == "1":
    mps_there = True

if fast_mode:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16, revision="fp16"
    )
    pipe.enable_attention_slicing()
    # Apply Token merging with a 50% merging ratio - this speeds up the model by ~ 2x
    tomesd.apply_patch(pipe)  # Can also use pipe.unet in place of pipe here

    # Check if Cuda is available before using deepspeed
    # Here, we only use deepspeed because it's apowerful speedup already, but we can optimize things further by creating a BATCH file that runs the file by calling Meta's xformers - https://github.com/facebookresearch/xformers
    # also use
    if cuda_there:
        with torch.inference_mode():
            deepspeed.init_inference(
                model=getattr(pipe, "model", pipe),
                mp_size=1,  # Number of GPUs used
                dtype=torch.float16,  # Data type of the weights
                replace_method="auto",  # How Deepspeed identifies which layers to replace
                replace_with_kernel_inject=False,  # Whether to replace the model with the kernel injector
            )
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_name)

# Scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# where to pipe?
if cuda_there:
    pipe = pipe.to("cuda")
elif mps_there:
    pipe = pipe.to("mps")
else:
    pipe.to("cpu")

# Number of inference steps
num_steps = 100


# Function to determine chessboard dimension based on parameter
def chessboard_dim(comparator):
    for i in range(2, comparator):
        acc = i
        new_acc = acc**3
        if new_acc > comparator:
            return acc
        acc = new_acc


# Function to generate the intrinsics.txt matrix from the image
def generate_intrinsics(view_scale, fovy, width, height):
    H = int(view_scale * height)
    W = int(view_scale * width)
    cx = H / 2
    cy = W / 2
    focal = H / (2 * np.tan(np.deg2rad(fovy) / 2))
    intrinsics = np.array([focal, focal, cx, cy])
    return intrinsics


# Function to generate pose
def generate_pose(radius):
    from scipy.spatial.transform import Rotation as R

    # Center point
    center = np.array([0, 0, 0], dtype=np.float32)
    # Rotation
    rotation = R.from_matrix(np.eye(3))
    # first move camera to radius
    res = np.eye(4, dtype=np.float32)
    res[2, 3] = radius
    # rotate
    rot = np.eye(4, dtype=np.float32)
    rot[:3, :3] = rotation.as_matrix()
    res = rot @ res
    # translate
    res[:3, 3] -= center
    return res


def generate_image(
    prompt,
    view_scale=1.5,
    fovy=20,
    radius=5,
    width=None,
    height=None,
    num_images=1,
    style=None,
):
    if not width:
        width = 512
    if not height:
        height = 512
    negative_prompt = (
        "An image with strongly assymetric measures and unrecognizable distortions."
    )
    image_folder = "data/"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    original_prompt = prompt
    if style:
        try:
            stylized_prompt = original_prompt + ", " + promptStyles.styles[style]
        except:
            print("Prompt style not found. Defaulting to prompt provided.")

    clean_prompt = original_prompt.replace(" ", "_").replace(".", ",")

    # Image subfolder generation
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # Generate pose
    pose = generate_pose(radius)
    image_pose = ""
    for row in pose:
        image_pose += " ".join([str(num) for num in row]) + "\n"

    # Generate intrinsics matrix containing focal lengths and center
    print("Generating intrinsics matrix")
    intrinsics = generate_intrinsics(view_scale, fovy, width, height)
    image_intrinsics = ""
    # Camera parameters
    image_intrinsics += " ".join([str(num) for num in intrinsics])

    # Other parameters by default 0 and 1
    image_intrinsics += "\n0. 0. 0.\n"
    image_intrinsics += "1.\n"

    # Width and height of the image
    dimensions = str(width) + " " + str(height) + "\n"
    image_intrinsics += dimensions

    # Generate hdf5 file
    print("Generating hdf5 file")
    hdf5_filename = image_folder + clean_prompt + ".hdf5"
    with h5py.File(hdf5_filename, "w") as file:
        # Generate the number of images specified
        for i in range(num_images):
            image = pipe(
                stylized_prompt,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                negative_prompt=negative_prompt,
            ).images[0]

            # Create a group in the file
            group = file.create_group("instance_" + str(i + 1))

            # Encode data
            img_array = np.array(image)
            image_pose = image_pose.encode("utf-8")
            image_intrinsics = image_intrinsics.encode("utf-8")

            # Create datasets for rgb, pose, intrinsics\
            group.create_dataset("rgb", data=img_array)
            group.create_dataset("pose", data=image_pose)
            group.create_dataset("intrinsics.txt", data=image_intrinsics)

    print("Images saved to " + image_folder)
    return image_folder
