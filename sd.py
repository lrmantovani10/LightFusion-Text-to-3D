import torch, os, promptStyles, tomesd
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from PIL import Image
import numpy as np
import h5py
import util
import cv2

model_name = "runwayml/stable-diffusion-v1-5"
controlnet_name = "fusing/stable-diffusion-v1-5-controlnet-openpose"
text_inversion_path = "textual_inversion/charturnerv2.pt"
poses_path = "textual_inversion/poses.png"
num_steps = 45
guidance_scale = 6.5


def build_pipe(device):
    controlnet = ControlNetModel.from_pretrained(
        controlnet_name, torch_dtype=torch.float32
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_name,
        controlnet=controlnet,
        torch_dtype=torch.float32,
    )

    pipe.load_textual_inversion(text_inversion_path)

    # Apply Token merging with a 50% merging ratio - this speeds up the model by ~ 2x
    tomesd.apply_patch(pipe)  # Can also use pipe.unet in place of pipe here

    # Check if Cuda is available before using deepspeed
    # Here, we only use deepspeed because it's a powerful speedup already, but we can optimize things further by creating a BATCH file that runs the code by calling Meta's xformers - https://github.com/facebookresearch/xformers
    # Alternatively, you can keep the code as is with the SDPA speedup already
    # implemented by default in torch >= 2.0. Check https://huggingface.co/docs/diffusers/optimization/torch2.0#benchmark for a comparison between xformers and SDPA
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
                replace_with_kernel_inject=False,
            )

    pipe.enable_attention_slicing()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if device == "mps":
        # Replacing MPS for CPU for now because MPS implementation is still in development -- change later if MPS version improves
        device = "cpu"
    print("Using " + str(device))
    pipe.to(device)
    if device == "cuda":
        # Currently only compiling the model for inference on CUDA. In the future, this might work well on MPS too
        pipe.enable_model_cpu_offload(device=device)
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    return pipe


def generate_intrinsics(width, height):
    """
    The camera intrinsics are represented in the following notation:
    (line 1) f (or fx), cx, cy
    (line 2) height, width

    Here, we use a camera approximation that has a 45 degree field of view and
    sensor size (in pixels) equal to twice the image width. Feel free to change these estimated values in your approximation.
    """
    cx = width / 2
    cy = height / 2
    fovy = 45
    sensor_size_estimation = 2 * width
    focal = sensor_size_estimation / (2 * np.tan(np.deg2rad(fovy) / 2))
    intrinsics = np.array([focal, cx, cy, height, width])

    return intrinsics


def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    Create a rotation matrix from Euler angles (roll, pitch, yaw).

    roll: Rotation angle around the x-axis in radians.
    pitch: Rotation angle around the y-axis in radians.
    yaw: Rotation angle around the z-axis in radians.
    returns aA 3x3 rotation matrix.
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


def generate_extrinsics(rotation):
    roll, pitch, yaw = rotation
    rotation_matrix = rotation_matrix_from_euler(roll, pitch, yaw)
    translation_vector = np.array([0, 0, 0])
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = translation_vector

    return pose_matrix


def generate_poses():
    poses = Image.open(poses_path)
    return poses


def generate_images(
    prompt,
    height=512,
    style=None,
    device="cuda",
    initial_negative_prompt=None,
    image_folder="image_data/",
    num_images=2,
):
    width = 4 * height
    pipe = build_pipe(device)

    util.cond_mkdir(image_folder)

    generated_images_folder = image_folder + "generated_images/"
    util.cond_mkdir(generated_images_folder)
    original_prompt = prompt.lower()

    prompt = (
        "(character sheet:1.6) of 1 "
        + prompt.lower()
        + " blank background, charturnerv2."
    )
    if style:
        try:
            prompt += ", " + promptStyles.styles[style]
        except:
            print("Prompt style not found. Defaulting to prompt provided.")

    clean_prompt = prompt.replace(" ", "_").replace(".", ",")
    original_prompt = original_prompt.replace(" ", "_").replace(".", ",")
    negative_prompt = (
        "((not full body)), small, gross proportions, bland colors, assymetric measures, unrecognizable distortions, deformed eyes, ((disfigured)), ((bad art)), ((deformed)), ((extra limbs)), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, poorly drawn eyes, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), cloned face, body out of frame, out of frame, bad anatomy, gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), (fused fingers), (too many fingers), (((long neck))), tiling, poorly drawn, mutated, cross-eye, canvas frame, frame, cartoon, 3d, weird colors"
        + ((", " + initial_negative_prompt) if initial_negative_prompt else "")
    )

    rotations = [
        # back
        (0, 0, 180),
        # 90 clockwise
        (0, 0, 270),
        # front
        (0, 0, 0),
        # 90 counterclockwise
        (0, 0, 90),
    ]

    gen_folder_num = len(
        [
            k
            for k in os.listdir(generated_images_folder)
            if ("png" in k.lower() or "jpeg" in k.lower() or "jpg" in k.lower())
        ]
    )

    poses = generate_poses()
    print("Final prompt: ", prompt)

    tiles_folder = image_folder + "individual_images/"
    util.cond_mkdir(tiles_folder)
    prompt_tile_folder = os.path.join(tiles_folder, clean_prompt)
    util.cond_mkdir(prompt_tile_folder)
    tiles_len = len(os.listdir(prompt_tile_folder))
    hdf5_filename_original = image_folder + original_prompt + ".hdf5"
    hdf5_filename_last = image_folder + original_prompt + "_generated.hdf5"
    hdf5_filename = hdf5_filename_original

    extrinsics = [generate_extrinsics(rotations[j]) for j in range(len(rotations))]
    final_width = int(width // num_poses)
    image_intrinsics = generate_intrinsics(final_width, height)

    # Check how many images with the same prompt have been generated before
    try:
        with h5py.File(hdf5_filename, "r") as file:
            num_equal_images = len(file.keys())
    except:
        num_equal_images = 0

    for i in range(num_images):
        if i == len(num_images) - 1:
            hdf5_filename = hdf5_filename_last
            num_equal_images = 0

        seed = generator.seed()
        print("Seed used: " + str(seed))
        generator = torch.Generator(device=device).manual_seed(seed)
        gen_folder_num += 1
        save_path = generated_images_folder + str(gen_folder_num) + ".png"

        latents = torch.randn(
            (1, pipe.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=device,
        )

        image = pipe(
            prompt,
            poses,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            negative_prompt=negative_prompt,
            latents=latents,
            guidance_scale=guidance_scale,
        ).images[0]

        image.save(save_path)
        img_array = np.array(image)

        # Extract four poses from the image
        img_arrays = []
        pose_arrays = []
        tiles_len += 1
        num_poses = len(rotations)
        tiles_folder_specific = os.path.join(tiles_folder, clean_prompt, str(tiles_len))
        util.cond_mkdir(tiles_folder_specific)
        for j in range(num_poses):
            # The first tile normally extends a bit further
            if j == 0:
                k = width / 30
            elif j == 1:
                k = width / 70
            else:
                k = 0
            effective_width = int((width // num_poses) + k)
            tile = img_array[
                :,
                j * effective_width : (j + 1) * effective_width,
                :,
            ]
            if j < 2:
                tile = cv2.resize(
                    tile,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            img_arrays.append(tile)
            pose_arrays.append(extrinsics[j])
            Image.fromarray(tile).save(tiles_folder_specific + str(j + 1) + ".png")

        num_equal_images += 1
        with h5py.File(hdf5_filename, "a") as file:
            group = file.create_group("instance_" + str(num_equal_images))
            group.create_dataset("intrinsics.txt", data=image_intrinsics)
            rgbs_data = group.create_group("rgb")
            poses_data = group.create_group("pose")

            for t in range(num_poses):
                poses_data.create_dataset(str(t + 1) + ".txt", data=pose_arrays[t])
                rgbs_data.create_dataset(str(t + 1) + ".png", data=img_arrays[t])

    print("Images saved to " + image_folder)
    return hdf5_filename_original
