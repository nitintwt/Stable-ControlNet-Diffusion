# This is the code I am running on my google colab to generate image

# !pip install --quiet --upgrade accelerate
# !pip install git+https://github.com/huggingface/diffusers.git@main
# !pip install controlnet_hinter==0.0.5

# we will be using controlnet to get the edges of the input image
import controlnet_hinter
# Mapping for ControlNet Model path and hinter function
CONTROLNET_MAPPING = {
    "canny_edge": {
        "model_id": "lllyasviel/sd-controlnet-canny",
        "hinter": controlnet_hinter.hint_canny
    },
    "pose": {
        "model_id": "lllyasviel/sd-controlnet-openpose",
        "hinter": controlnet_hinter.hint_openpose
    },
    "depth": {
        "model_id": "lllyasviel/sd-controlnet-depth",
        "hinter": controlnet_hinter.hint_depth
    },
    "scribble": {
        "model_id": "lllyasviel/sd-controlnet-scribble",
        "hinter": controlnet_hinter.hint_scribble,
    },
    "segmentation": {
        "model_id": "lllyasviel/sd-controlnet-seg",
        "hinter": controlnet_hinter.hint_segmentation,
    },
    "normal": {
        "model_id": "lllyasviel/sd-controlnet-normal",
        "hinter": controlnet_hinter.hint_normal,
    },
    "hed": {
        "model_id": "lllyasviel/sd-controlnet-hed",
        "hinter": controlnet_hinter.hint_hed,
    },
    "hough": {
        "model_id": "lllyasviel/sd-controlnet-mlsd",
        "hinter": controlnet_hinter.hint_hough,
    }
}


from diffusers import DiffusionPipeline, LCMScheduler,DDIMScheduler
from diffusers import AutoPipelineForText2Image, StableDiffusionControlNetPipeline, ControlNetModel
import torch
from diffusers.utils import load_image


# Stable diffusion base model
base_model_path =  "digiplay/Juggernaut_final"

pipe=None
torch.cuda.empty_cache()
device = "cuda"

controlnet_type = "canny_edge"  # model to get edges

# merging the Stable diffusion base model with the ControlNet model using diffusers pipeline
controlnet = ControlNetModel.from_pretrained(CONTROLNET_MAPPING[controlnet_type]["model_id"], torch_dtype=torch.float16).to(device)
pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path,controlnet=controlnet,torch_dtype=torch.float16).to(device)


logo_image_url = "https://upload.wikimedia.org/wikipedia/commons/3/36/United_Airlines_Boeing_777-200_Meulemans.jpg"
logo_image = load_image(logo_image_url)


# Describe the prompt for the new image
prompt = "water , sea , aquatic life, natural, detailed, hd, 4k, best quality, extremely detailed"
negative_prompt = "nsfw, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"


no_of_steps = 20

# how much would the prompt affect the final output.
# higher guidance scale means more preference given to the prompt.
guidace_scale = 7.0

# how much final output would follow the control image
controlnet_conditioning_scale=1.0

# convert input image into a control image based on the model
control_image = CONTROLNET_MAPPING[controlnet_type]["hinter"](logo_image )

# run the inference,
my_images = pipe(
    prompt=prompt,
    width=512,
    height=512,
    negative_prompt=negative_prompt,
    image=control_image,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    num_inference_steps=no_of_steps,
    guidance_scale=guidace_scale,
)

# get first image from the image generations object.
first_image = my_images.images[0]

first_image