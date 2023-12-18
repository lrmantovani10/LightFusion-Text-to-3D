import torch, os, promptStyles, tomesd
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import numpy as np
import h5py
import util

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model = "stabilityai/stable-diffusion-xl-refiner-1.0"


def build_pipes(device):
    pipe_base = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        revision="main",
    )

    pipe_refiner = StableDiffusionXLPipeline.from_pretrained(
        refiner_model,
        torch_dtype=torch.float32,
        revision="main",
    )

    for pipe in [pipe_base, pipe_refiner]:
        pipe.enable_attention_slicing()
        # Apply Token merging with a 50% merging ratio - this speeds up the model by ~ 2x
        tomesd.apply_patch(pipe)  # Can also use pipe.unet in place of pipe here

        # Check if Cuda is available before using deepspeed
        # Here, we only use deepspeed because it's a powerful speedup already, but we can optimize things further by creating a BATCH file that runs the code by calling Meta's xformers - https://github.com/facebookresearch/xformers
        if device == "cuda":
            # If there are any problems installing deepspeed, run "pip install py-cpuinfo" before installing deepspeed
            # For MPS to work, you must (as of this writing) use the PyTorch nightly build, so make sure that is the one you have installed for your virtual env
            import deepspeed

            with torch.inference_mode():
                deepspeed.init_inference(
                    model=getattr(pipe, "model", pipe),
                    mp_size=1,  # Number of GPUs used
                    dtype=torch.float16,  # Data type of the weights
                    replace_method="auto",  # How Deepspeed identifies which layers to replace
                    replace_with_kernel_inject=False,  # Whether to replace the model with the kernel injector
                )

        # Scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # where to pipe?
        if device == "cuda":
            print("Using CUDA")
            pipe = pipe.to("cuda")
        elif device == "mps":
            print("Using MPS")
            # Uncomment this to use MPS on a Mac instead of CPU (may run into memory shortage issues, but if there is enough memory that can be allocated, will be much faster than CPU)
            # pipe = pipe.to("mps")
        else:
            print("Using CPU")
            pipe.to("cpu")

    return pipe_base, pipe_refiner


# Number of inference steps
num_steps = 100


# Function to generate the camera intrinsics parameters
def generate_intrinsics(fovy, width, height):
    """
    The camera intrinsics are represented in the following notation:
    (line 1) f (or fx), cx, cy
    (line 2) height, width
    """
    cx = width / 2
    cy = height / 2
    focal = width / (2 * np.tan(np.deg2rad(fovy) / 2))
    intrinsics = np.array([focal, cx, cy])
    image_intrinsics = " ".join([str(num) for num in intrinsics])
    image_intrinsics += str(height) + " " + str(width) + "\n"
    image_intrinsics = image_intrinsics.encode("utf-8")
    return image_intrinsics


def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    Create a rotation matrix from Euler angles (roll, pitch, yaw).

    :param roll: Rotation angle around the x-axis in radians.
    :param pitch: Rotation angle around the y-axis in radians.
    :param yaw: Rotation angle around the z-axis in radians.
    :return: A 3x3 rotation matrix.
    """
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# Function to generate camera extrinsics parameters
def generate_extrinsics(rotation):
    roll, pitch, yaw = rotation
    rotation_matrix = rotation_matrix_from_euler(roll, pitch, yaw)
    translation_vector = np.array([0, 0, 0])
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = translation_vector

    image_pose = ""
    for row in pose_matrix:
        image_pose += " ".join([str(num) for num in row]) + "\n"
    image_pose = image_pose.encode("utf-8")

    return image_pose


def generate_images(
    prompt,
    fovy=45,
    width=None,
    height=None,
    style=None,
    device="cuda",
    initial_negative_prompt=None,
):
    pipe_base, pipe_refiner = build_pipes(device)

    if not width:
        width = 768
    if not height:
        height = 768

    final_width = final_height = 128
    image_folder = "image_data/"
    util.cond_mkdir(image_folder)

    generated_images_folder = image_folder + "generated_images/"
    util.cond_mkdir(generated_images_folder)

    prompt += ", in the center of a blank background"
    if style:
        try:
            prompt += ", " + promptStyles.styles[style]
        except:
            print("Prompt style not found. Defaulting to prompt provided.")

    clean_prompt = prompt.replace(" ", "_").replace(".", ",")
    negative_prompt = (
        "not photorealistic, assymetric measures, unrecognizable distortions, blurry, disfigured, small, not full body, no vibarant colors, not blank background, toy, not at the center of the image, deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ((((mutated hands and fingers)))), (((out of frame))), cartoon, 3d, (disfigured), (bad art), (deformed), (poorly drawn), (extra limbs), strange colours, blurry, boring, sketch, lacklustre, repetitive, cropped, hands"
        + ((", " + initial_negative_prompt) if initial_negative_prompt else "")
    )

    poses = [
        "front",
        "90 degrees clockwise",
        "90 degrees counterclockwise",
        "back",
        "overhead",
        "bottom",
    ]
    rotations = [
        (0, 0, 0),
        (0, 0, 270),
        (0, 0, 90),
        (0, 0, 180),
        (0, 270, 0),
        (0, 90, 0),
    ]

    first_image = None
    hdf5_filename = image_folder + clean_prompt + ".hdf5"

    # Check how many images with the same prompt have been generated before
    try:
        with h5py.File(hdf5_filename, "r") as file:
            num_equal_images = len(file.keys())
    except:
        num_equal_images = 0

    # Latent generation with a fixed seed so that we ensure that
    # the generated images are similar
    generator = torch.Generator(device=device)
    # For a random seed, run the code below. The one I am using works well for a single image generation with a blank background
    # seed = generator.seed()
    seed = 1500722359
    print("Seed used: " + str(seed))
    generator = generator.manual_seed(seed)

    img_arrays = []
    pose_arrays = []
    gen_folder_len = len(os.listdir(generated_images_folder))

    for i, pose in enumerate(poses):
        if first_image is None:
            # Include pose in propmpt
            pose_prompt = (
                prompt
                + (", viewed  from the " if i < 1 or i > 2 else ", viewed rotated ")
                + pose
            )

            latents = torch.randn(
                (1, pipe_base.unet.config.in_channels, height // 8, width // 8),
                generator=generator,
                device=device,
            )

            image = pipe_base(
                pose_prompt,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                negative_prompt=negative_prompt,
                latents=latents,
            ).images[0]

            first_image = np.array(image)

        else:
            pose_prompt = (
                "The same "
                + prompt
                + " as in this image, but viewed from the "
                + pose
                + " position."
            )

            latents = torch.randn(
                (1, pipe_refiner.unet.config.in_channels, height // 8, width // 8),
                generator=generator,
                device=device,
            )

            # Non-frontal images do not include a face
            pose_negative_prompt = negative_prompt + ", includes a face."
            image = pipe_refiner(
                pose_prompt,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                negative_prompt=pose_negative_prompt,
                init_image=first_image,
                latents=latents,
            ).images[0]

        gen_folder_len += 1
        image.save(generated_images_folder + str(gen_folder_len) + ".png")
        image = image.resize((final_width, final_height))
        img_array = np.array(image).encode("utf-8")

        image_pose = generate_extrinsics(rotations[i])

        img_arrays.append(img_array)
        pose_arrays.append(image_pose)

    with h5py.File(hdf5_filename, "a") as file:
        group = file.create_group("instance_" + str(num_equal_images + 1))

        print("Generating camera intrinsics")
        image_intrinsics = generate_intrinsics(fovy, width, height)

        rgbs_data = group.create_group("rgb")
        poses_data = group.create_group("pose")
        group.create_dataset("intrinsics.txt", data=image_intrinsics)

        for i in range(len(poses)):
            poses_data.create_dataset(str(i + 1) + ".png", data=pose_arrays[i])
            rgbs_data.create_dataset(str(i + 1) + ".txt", data=img_arrays[i])

        print("Images saved to " + image_folder)
        return hdf5_filename
