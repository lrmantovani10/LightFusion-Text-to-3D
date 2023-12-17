import torch, os, promptStyles, tomesd
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import numpy as np
import h5py

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Fast mode - slightly lower quality tradeoff
fast_mode = False

# Check if CUDA is available
cuda_there = False
# Check if MPS is available
mps_there = False
if torch.cuda.is_available():
    # If there are any problems installing deepspeed, run "pip install py-cpuinfo" before installing deepspeed
    import deepspeed

    cuda_there = True

# For MPS to work, you must (as of this writing) use the PyTorch nightly build, so make sure that is the one you have installed for your virtual env
elif torch.backends.mps.is_available():
    mps_there = True

pipe_base = StableDiffusionXLPipeline.from_pretrained(
    base_model,
    torch_dtype=(torch.float16 if fast_mode else torch.float32),
    revision="main",
)

pipe_refiner = StableDiffusionXLPipeline.from_pretrained(
    refiner_model,
    torch_dtype=(torch.float16 if fast_mode else torch.float32),
    revision="main",
)

if fast_mode:
    for pipe in [pipe_base, pipe_refiner]:
        pipe.enable_attention_slicing()
        # Apply Token merging with a 50% merging ratio - this speeds up the model by ~ 2x
        tomesd.apply_patch(pipe)  # Can also use pipe.unet in place of pipe here

        # Check if Cuda is available before using deepspeed
        # Here, we only use deepspeed because it's a powerful speedup already, but we can optimize things further by creating a BATCH file that runs the code by calling Meta's xformers - https://github.com/facebookresearch/xformers
        if cuda_there:
            with torch.inference_mode():
                deepspeed.init_inference(
                    model=getattr(pipe, "model", pipe),
                    mp_size=1,  # Number of GPUs used
                    dtype=torch.float16,  # Data type of the weights
                    replace_method="auto",  # How Deepspeed identifies which layers to replace
                    replace_with_kernel_inject=False,  # Whether to replace the model with the kernel injector
                )

for pipe in [pipe_base, pipe_refiner]:
    # Scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # where to pipe?
    if cuda_there:
        print("Using CUDA")
        pipe = pipe.to("cuda")
    elif mps_there:
        print("Using MPS")
        pipe = pipe.to("mps")
    else:
        print("Using CPU")
        pipe.to("cpu")

# Number of inference steps
num_steps = 100


# Function to generate the intrinsics.txt matrix from the image
def generate_intrinsics(view_scale, fovy, width, height):
    H = int(view_scale * height)
    W = int(view_scale * width)
    cx = H / 2
    cy = W / 2
    focal = H / (2 * np.tan(np.deg2rad(fovy) / 2))
    intrinsics = np.array([focal, focal, cx, cy])
    return intrinsics


# Convert degrees to radians
def deg_to_rad(deg):
    return deg * np.pi / 180


# Define the rotation matrices for rotations around the X and Y axes
def rotation_x(angle):
    rad = deg_to_rad(angle)
    return np.array(
        [[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]]
    )


def rotation_y(angle):
    rad = deg_to_rad(angle)
    return np.array(
        [[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]]
    )


# Function to generate pose
def generate_pose_data(rotation):
    # Image rotations
    x_rotation = rotation_x(rotation[0])
    y_rotation = rotation_y(rotation[1])

    # Rotation matrix
    rotation_matrix = np.dot(y_rotation, x_rotation)

    # Translation vector
    translation_vector = np.array([0, 0, 0])

    # Constructing the 4x4 pose matrix
    pose_matrix = np.eye(4)  # Start with an identity matrix
    pose_matrix[:3, :3] = rotation_matrix  # Insert rotation
    pose_matrix[:3, 3] = translation_vector  # Insert translation

    return pose_matrix


# Function to generate images using SD
def generate_images(
    prompt,
    view_scale=1.5,
    fovy=20,
    radius=5,
    width=None,
    height=None,
    style=None,
    device="cuda",
    initial_negative_prompt=None,
):
    # Define image dimensions
    if not width:
        width = 768
    if not height:
        height = 768

    final_width = final_height = 128

    # Define the image folder
    image_folder = "image_data/"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    generated_images_folder = image_folder + "generated_images/"
    if not os.path.exists(generated_images_folder):
        os.makedirs(generated_images_folder)

    prompt += ", in the center of a blank background, only a single figure"
    # Add style to prompt
    if style:
        try:
            prompt += ", " + promptStyles.styles[style]
        except:
            print("Prompt style not found. Defaulting to prompt provided.")

    clean_prompt = prompt.replace(" ", "_").replace(".", ",")

    # What we do not want to see
    negative_prompt = (
        "assymetric measures, unrecognizable distortions, blurry, disfigured, multiple objects, not full body, no color, non blank background, non photorealistic, toy, deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ((((mutated hands and fingers)))), (((out of frame))), cartoon, 3d, (disfigured), (bad art), (deformed), (poorly drawn), (extra limbs), strange colours, blurry, boring, sketch, lacklustre, repetitive, cropped, hands"
        + ((", " + initial_negative_prompt) if initial_negative_prompt else "")
    )

    # Generate intrinsics matrix containing focal lengths and center
    print("Generating intrinsics matrix")
    intrinsics = generate_intrinsics(view_scale, fovy, width, height)
    image_intrinsics = " ".join([str(num) for num in intrinsics])
    # Other parameters by default 0 and 1
    image_intrinsics += "\n0. 0. 0.\n"
    image_intrinsics += "1.\n"
    dimensions = str(width) + " " + str(height) + "\n"
    image_intrinsics += dimensions
    image_intrinsics = image_intrinsics.encode("utf-8")

    # Pose names and corresponding rotations
    poses = ["front", "right sideview", "back", "left sideview", "overhead", "bottom"]
    rotations = [(0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, 270)]

    # First image generated
    first_image = None

    # Generate hdf5 file
    print("Generating hdf5 file")
    hdf5_filename = image_folder + clean_prompt + ".hdf5"

    # Check how many images with the same prompt have been generated before
    num_equal_images = 0
    try:
        with h5py.File(hdf5_filename, "r") as file:
            num_equal_images = len(file.keys())
    except:
        pass

    # Latent generation with a fixed seed so that we ensure that
    # the generated images are similar
    generator = torch.Generator(device=device)
    seed = generator.seed()
    print("Seed used: " + str(seed))
    generator = generator.manual_seed(seed)

    img_arrays = []
    pose_arrays = []
    intrinsics_arrays = []

    # Generate images for each pose
    for i, pose in enumerate(poses):
        pose_data = generate_pose_data(rotations[i])
        image_pose = ""
        for row in pose_data:
            image_pose += " ".join([str(num) for num in row]) + "\n"

        if not first_image:
            # Include pose in propmpt
            pose_prompt = prompt + ", viewed from the " + pose + " position."

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
            first_image = image

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

        # Save the images in the generated_images folder
        image.save(
            generated_images_folder
            + str(len(os.listdir(generated_images_folder)) + 1)
            + ".png"
        )

        # Encode the data
        image = image.resize((final_width, final_height))
        img_array = np.array(image)
        image_pose = image_pose.encode("utf-8")

        # Append the data to the arrays
        img_arrays.append(img_array)
        pose_arrays.append(image_pose)
        intrinsics_arrays.append(image_intrinsics)

    # Save the images in the hdf5 file
    with h5py.File(hdf5_filename, "a") as file:
        # Create a group in the file
        group = file.create_group("instance_" + str(num_equal_images + 1))

        # Create datasets for rgb, pose, intrinsics
        rgbs_data = group.create_group("rgb")
        poses_data = group.create_group("pose")
        group.create_dataset("intrinsics.txt", data=image_intrinsics)

        for i in range(len(poses)):
            poses_data.create_dataset(str(i + 1) + ".png", data=pose_arrays[i])
            rgbs_data.create_dataset(str(i + 1) + ".txt", data=img_arrays[i])

        print("Images saved to " + image_folder)
        return hdf5_filename
