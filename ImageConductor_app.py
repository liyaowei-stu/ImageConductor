import os
import sys


import numpy as np
import cv2
import uuid
import torch
import torchvision
import json

from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange, repeat
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler

from pipelines.pipeline_imagecoductor import ImageConductorPipeline
from modules.unet import UNet3DConditionFlowModel
from utils.gradio_utils import ensure_dirname, split_filename, visualize_drag, image2pil, image2arr
from utils.utils import create_image_controlnet, create_flow_controlnet, interpolate_trajectory, load_weights, load_model, bivariate_Gaussian
from utils.lora_utils import add_LoRA_to_controlnet
from utils.visualizer import Visualizer, vis_flow_to_video

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(current_dir, 'gradio'))
import gradio as gr

#### Description ####
title = r"""<h1 align="center">CustomNet: Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models</h1>"""

head = r"""
<div style="text-align: center;">
                        <h1>Image Conductor: Precision Control for Interactive Video Synthesis</h1>
                        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                            <a href=""></a>
                            <a href='https://liyaowei-stu.github.io/project/ImageConductor/'><img src='https://img.shields.io/badge/Project_Page-ImgaeConductor-green' alt='Project Page'></a>
                            <a href='https://arxiv.org/pdf/2406.15339'><img src='https://img.shields.io/badge/Paper-Arxiv-blue'></a>
                            <a href='https://github.com/liyaowei-stu/ImageConductor'><img src='https://img.shields.io/badge/Code-Github-orange'></a>
                            

                        </div>
                        </br>
</div>
"""



descriptions = r"""
Official Gradio Demo for <a href='https://github.com/liyaowei-stu/ImageConductor'><b>Image Conductor: Precision Control for Interactive Video Synthesis</b></a>.<br>
🧙Image Conductor enables precise, fine-grained control for generating motion-controllable videos from images, advancing the practical application of interactive video synthesis.<br>
"""


instructions = r"""
            - ⭐️ <b>step1: </b>Upload or select one image from Example.
            - ⭐️ <b>step2: </b>Click 'Add Drag' to draw some drags.
            - ⭐️ <b>step3: </b>Input text prompt  that complements the image (Necessary).
            - ⭐️ <b>step4: </b>Select 'Drag Mode' to specify the control of camera transition or object movement.
            - ⭐️ <b>step5: </b>Click 'Run' button to generate video assets.
            - ⭐️ <b>others: </b>Click 'Delete last drag' to delete the whole lastest path. Click 'Delete last step' to delete the lastest clicked control point.
            """

citation = r"""
If Image Conductor is helpful, please help to ⭐ the <a href='https://github.com/liyaowei-stu/ImageConductor' target='_blank'>Github Repo</a>. Thanks! 
[![GitHub Stars](https://img.shields.io/github/stars/liyaowei-stu%2FImageConductor)](https://github.com/liyaowei-stu/ImageConductor)
---

📝 **Citation**
<br>
If our work is useful for your research, please consider citing:
```bibtex
@misc{li2024imageconductor,
    title={Image Conductor: Precision Control for Interactive Video Synthesis}, 
    author={Li, Yaowei and Wang, Xintao and Zhang, Zhaoyang and Wang, Zhouxia and Yuan, Ziyang and Xie, Liangbin and Zou, Yuexian and Shan, Ying},
    year={2024},
    eprint={2406.15339},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>ywl@stu.pku.edu.cn</b>.

# """

# os.makedirs("models/personalized")
# os.makedirs("models/sd1-5")

# if not os.path.exists("models/flow_controlnet.ckpt"):
#     os.system(f'wget https://huggingface.co/TencentARC/ImageConductor/resolve/main/flow_controlnet.ckpt?download=true -P models/')
#     os.system(f'mv models/flow_controlnet.ckpt?download=true models/flow_controlnet.ckpt')

# if not os.path.exists("models/image_controlnet.ckpt"):
#     os.system(f'wget https://huggingface.co/TencentARC/ImageConductor/resolve/main/image_controlnet.ckpt?download=true -P models/')
#     os.system(f'mv models/image_controlnet.ckpt?download=true models/image_controlnet.ckpt')


# if not os.path.exists("models/unet.ckpt"):
#     os.system(f'wget https://huggingface.co/TencentARC/ImageConductor/resolve/main/unet.ckpt?download=true -P models/')
#     os.system(f'mv models/unet.ckpt?download=true models/unet.ckpt')

# if not os.path.exists("models/sd1-5/config.json"):
#     os.system(f'wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/config.json?download=true -P models/sd1-5/')
#     os.system(f'mv models/sd1-5/config.json?download=true  models/sd1-5/config.json')

# if not os.path.exists("models/sd1-5/unet.ckpt"):
#     os.system(f'cp -r models/unet.ckpt  models/sd1-5/unet.ckpt')

# # os.system(f'wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin?download=true -P models/sd1-5/')

# if not os.path.exists("models/personalized/helloobjects_V12c.safetensors"):
#     os.system(f'wget https://huggingface.co/TencentARC/ImageConductor/resolve/main/helloobjects_V12c.safetensors?download=true -P models/personalized')
#     os.system(f'mv models/personalized/helloobjects_V12c.safetensors?download=true models/personalized/helloobjects_V12c.safetensors')


# if not os.path.exists("models/personalized/TUSUN.safetensors"):
#     os.system(f'wget https://huggingface.co/TencentARC/ImageConductor/resolve/main/TUSUN.safetensors?download=true -P models/personalized')
#     os.system(f'mv models/personalized/TUSUN.safetensors?download=true models/personalized/TUSUN.safetensors')



# - - - - - examples  - - - - -  #

image_examples = [
    ["__asset__/images/object/turtle-1.jpg", 
     "a sea turtle gracefully swimming over a coral reef in the clear blue ocean.", 
     "object",
     11318446767408804497,
     "",
     "turtle"
    #  "__asset__/turtle.mp4",
     ],
    
    ["__asset__/images/object/rose-1.jpg", 
     "a red rose engulfed in flames.", 
     "object",
     6854275249656120509,
     "",
     "rose",
    #  "__asset__/rose.mp4"
     ],
    
    ["__asset__/images/object/jellyfish-1.jpg", 
     "intricate detailing,photorealism,hyperrealistic, glowing jellyfish mushroom, flying, starry sky, bokeh, golden ratio composition.", 
     "object",
     17966188172968903484,
     "HelloObject",
     "jellyfish",
     ],
    
    
    ["__asset__/images/camera/lush-1.jpg", 
     "detailed craftsmanship, photorealism, hyperrealistic, roaring waterfall, misty spray, lush greenery, vibrant rainbow, golden ratio composition.", 
     "camera",
     7970487946960948963,
     "HelloObject",
     "lush",
    #  "__asset__/lush.mp4",
     ],
    
    ["__asset__/images/camera/tusun-1.jpg", 
     "tusuncub with its mouth open, blurry, open mouth, fangs, photo background, looking at viewer, tongue, full body, solo, cute and lovely, Beautiful and realistic eye details, perfect anatomy, Nonsense, pure background, Centered-Shot, realistic photo, photograph, 4k, hyper detailed, DSLR, 24 Megapixels, 8mm Lens, Full Frame, film grain, Global Illumination, studio Lighting, Award Winning Photography, diffuse reflection, ray tracing.", 
     "camera",
     15131888710792130110,
     "TUSUN",
     "tusun",
     ],
    
    ["__asset__/images/camera/painting-1.jpg", 
     "A oil painting.", 
     "camera",
     42,
     "",
     "painting",
     ],
]


POINTS = {
    'turtle': "__asset__/trajs/object/turtle-1.json",
    'rose': "__asset__/trajs/object/rose-1.json",
    'jellyfish': "__asset__/trajs/object/jellyfish-1.json",
    'lush': "__asset__/trajs/camera/lush-1.json",
    'tusun': "__asset__/trajs/camera/tusun-1.json",
    'painting': "__asset__/trajs/camera/painting-1.json",
}

IMAGE_PATH = {
    'turtle': "__asset__/images/object/turtle-1.jpg",
    'rose': "__asset__/images/object/rose-1.jpg",
    'jellyfish': "__asset__/images/object/jellyfish-1.jpg",
    'lush': "__asset__/images/camera/lush-1.jpg",
    'tusun': "__asset__/images/camera/tusun-1.jpg",
    'painting': "__asset__/images/camera/painting-1.jpg",
}



DREAM_BOOTH = {
    'HelloObject': 'models/personalized/helloobjects_V12c.safetensors',
}

LORA = {
    'TUSUN': 'models/personalized/TUSUN.safetensors',
}

LORA_ALPHA = {
    'TUSUN': 0.6,
}

NPROMPT = {
    "HelloObject": 'FastNegativeV2,(bad-artist:1),(worst quality, low quality:1.4),(bad_prompt_version2:0.8),bad-hands-5,lowres,bad anatomy,bad hands,((text)),(watermark),error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,((username)),blurry,(extra limbs),bad-artist-anime,badhandv4,EasyNegative,ng_deepnegative_v1_75t,verybadimagenegative_v1.3,BadDream,(three hands:1.6),(three legs:1.2),(more than two hands:1.4),(more than two legs,:1.2)'    
}

output_dir = "outputs"
ensure_dirname(output_dir)

def points_to_flows(track_points, model_length, height, width):
    input_drag = np.zeros((model_length - 1, height, width, 2))
    for splited_track in track_points:
        if len(splited_track) == 1: # stationary point
            displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
            splited_track = tuple([splited_track[0], displacement_point])
        # interpolate the track
        splited_track = interpolate_trajectory(splited_track, model_length)
        splited_track = splited_track[:model_length]
        if len(splited_track) < model_length:
            splited_track = splited_track + [splited_track[-1]] * (model_length -len(splited_track))
        for i in range(model_length - 1):
            start_point = splited_track[i]
            end_point = splited_track[i+1]
            input_drag[i][int(start_point[1])][int(start_point[0])][0] = end_point[0] - start_point[0]
            input_drag[i][int(start_point[1])][int(start_point[0])][1] = end_point[1] - start_point[1]
    return input_drag

class ImageConductor:
    def __init__(self, device, unet_path, image_controlnet_path, flow_controlnet_path, height, width, model_length, lora_rank=64):
        self.device = device
        tokenizer    = CLIPTokenizer.from_pretrained("models/stable-diffusion-v1-5", subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained("models/stable-diffusion-v1-5", subfolder="text_encoder").to(device)
        vae          = AutoencoderKL.from_pretrained("models/stable-diffusion-v1-5", subfolder="vae").to(device)
        inference_config = OmegaConf.load("configs/inference/inference.yaml")
        unet = UNet3DConditionFlowModel.from_pretrained_2d("models/stable-diffusion-v1-5", subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

        self.vae = vae

        ### >>> Initialize UNet module >>> ###
        load_model(unet, unet_path)

        ### >>> Initialize image controlnet module >>> ###
        image_controlnet = create_image_controlnet("configs/inference/image_condition.yaml", unet)
        load_model(image_controlnet, image_controlnet_path)
        ### >>> Initialize flow controlnet module >>> ###
        flow_controlnet = create_flow_controlnet("configs/inference/flow_condition.yaml", unet)
        add_LoRA_to_controlnet(lora_rank, flow_controlnet)
        load_model(flow_controlnet, flow_controlnet_path)

        unet.eval().to(device)
        image_controlnet.eval().to(device)
        flow_controlnet.eval().to(device)

        self.pipeline = ImageConductorPipeline(
            unet=unet,
            vae=vae, 
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            image_controlnet=image_controlnet,
            flow_controlnet=flow_controlnet,
        ).to(device)

        
        self.height = height
        self.width = width
        # _, model_step, _ = split_filename(model_path)
        # self.ouput_prefix = f'{model_step}_{width}X{height}'
        self.model_length = model_length

        blur_kernel = bivariate_Gaussian(kernel_size=99, sig_x=10, sig_y=10, theta=0, grid=None, isotropic=True)

        self.blur_kernel = blur_kernel

    @torch.no_grad()
    def run(self, first_frame_path, tracking_points, prompt, drag_mode, negative_prompt, seed, randomize_seed, guidance_scale, num_inference_steps, personalized, examples_type):
        if examples_type != "":
            ### for adapting high version gradio
            first_frame_path = IMAGE_PATH[examples_type]
            tracking_points = json.load(open(POINTS[examples_type]))
            print("example first_frame_path", first_frame_path)
            print("example tracking_points", tracking_points)
        
        original_width, original_height=384, 256
        if isinstance(tracking_points, list):
            input_all_points = tracking_points
        else:
            input_all_points = tracking_points.value
        
        
        resized_all_points = [tuple([tuple([float(e1[0]*self.width/original_width), float(e1[1]*self.height/original_height)]) for e1 in e]) for e in input_all_points]

        dir, base, ext = split_filename(first_frame_path)
        id = base.split('_')[-1]
        
        
        # with open(f'{output_dir}/points-{id}.json', 'w') as f:
        #     json.dump(input_all_points, f)
        
        
        visualized_drag, _ = visualize_drag(first_frame_path, resized_all_points, drag_mode, self.width, self.height, self.model_length)

        ## image condition        
        image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (self.height, self.width), (1.0, 1.0), 
                    ratio=(self.width/self.height, self.width/self.height)
                ),
                transforms.ToTensor(),
            ])

        image_norm = lambda x: x
        image_paths = [first_frame_path]
        controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]
        controlnet_images = torch.stack(controlnet_images).unsqueeze(0).to(device)
        controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")
        num_controlnet_images = controlnet_images.shape[2]
        controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
        self.vae.to(device)
        controlnet_images = self.vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
        controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # flow condition
        controlnet_flows = points_to_flows(resized_all_points, self.model_length, self.height, self.width)
        for i in range(0, self.model_length-1):
            controlnet_flows[i] = cv2.filter2D(controlnet_flows[i], -1, self.blur_kernel)
        controlnet_flows = np.concatenate([np.zeros_like(controlnet_flows[0])[np.newaxis, ...], controlnet_flows], axis=0)  # pad the first frame with zero flow
        os.makedirs(os.path.join(output_dir, "control_flows"), exist_ok=True)
        trajs_video = vis_flow_to_video(controlnet_flows, num_frames=self.model_length) # T-1 x H x W x 3
        torchvision.io.write_video(f'{output_dir}/control_flows/sample-{id}-train_flow.mp4', trajs_video, fps=8, video_codec='h264', options={'crf': '10'})
        controlnet_flows = torch.from_numpy(controlnet_flows)[None][:, :self.model_length, ...]
        controlnet_flows =  rearrange(controlnet_flows, "b f h w c-> b c f h w").float().to(device)

        dreambooth_model_path = DREAM_BOOTH.get(personalized, '')
        lora_model_path = LORA.get(personalized, '')
        lora_alpha = LORA_ALPHA.get(personalized, 0.6)
        self.pipeline = load_weights(
            self.pipeline,
            dreambooth_model_path      = dreambooth_model_path,
            lora_model_path            = lora_model_path,
            lora_alpha                 = lora_alpha,
        ).to(device)
        
        if NPROMPT.get(personalized, '') != '':
            negative_prompt =  NPROMPT.get(personalized)
        
        if randomize_seed:
            random_seed = torch.seed()
        else:
            seed = int(seed)
            random_seed = seed
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  
        print(f"current seed: {torch.initial_seed()}")
        sample = self.pipeline(
                    prompt,
                    negative_prompt     = negative_prompt,
                    num_inference_steps = num_inference_steps,
                    guidance_scale      = guidance_scale,
                    width               = self.width,
                    height              = self.height,
                    video_length        = self.model_length,
                    controlnet_images = controlnet_images, # 1 4 1 32 48
                    controlnet_image_index = [0], 
                    controlnet_flows  = controlnet_flows,# [1, 2, 16, 256, 384]
                    control_mode = drag_mode,
                    eval_mode = True,
                ).videos

        outputs_path = os.path.join(output_dir, f'output_{i}_{id}.mp4')
        vis_video = (rearrange(sample[0], 'c t h w -> t h w c') * 255.).clip(0, 255)
        torchvision.io.write_video(outputs_path, vis_video, fps=8, video_codec='h264', options={'crf': '10'})

        return {output_image: visualized_drag, output_video: outputs_path}


def reset_states(first_frame_path, tracking_points):
    first_frame_path = gr.State()
    tracking_points = gr.State([])
    return {input_image:None, first_frame_path_var: first_frame_path, tracking_points_var: tracking_points}


def preprocess_image(image, tracking_points):
    image_pil = image2pil(image.name)
    raw_w, raw_h = image_pil.size
    resize_ratio = max(384/raw_w, 256/raw_h)
    image_pil = image_pil.resize((int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR)
    image_pil = transforms.CenterCrop((256, 384))(image_pil.convert('RGB'))
    id = str(uuid.uuid4())[:4]
    first_frame_path = os.path.join(output_dir, f"first_frame_{id}.jpg")
    image_pil.save(first_frame_path, quality=95)
    return {input_image: first_frame_path, first_frame_path_var: first_frame_path, tracking_points_var: gr.State([]), personalized: ""}


def add_tracking_points(tracking_points, first_frame_path, drag_mode, evt: gr.SelectData):  # SelectData is a subclass of EventData
    if drag_mode=='object':
        color = (255, 0, 0, 255)
    elif drag_mode=='camera':
        color = (0, 0, 255, 255)

    print(f"You selected {evt.value} at {evt.index} from {evt.target}")
    tracking_points.value[-1].append(evt.index)
    print(tracking_points.value)
    
    transparent_background = Image.open(first_frame_path).convert('RGBA')
    w, h = transparent_background.size
    transparent_layer = np.zeros((h, w, 4))
    for track in tracking_points.value:
        if len(track) > 1:
            for i in range(len(track)-1):
                start_point = track[i]
                end_point = track[i+1]
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(track)-2:
                    cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), color, 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, tuple(start_point), tuple(end_point), color, 2,)
        else:
            cv2.circle(transparent_layer, tuple(track[0]), 5, color, -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    return {tracking_points_var: tracking_points, input_image: trajectory_map}


def add_drag(tracking_points):
    tracking_points.value.append([])
    print(tracking_points.value)
    return {tracking_points_var: tracking_points}
    

def delete_last_drag(tracking_points, first_frame_path, drag_mode):
    if drag_mode=='object':
        color = (255, 0, 0, 255)
    elif drag_mode=='camera':
        color = (0, 0, 255, 255)
    tracking_points.value.pop()
    transparent_background = Image.open(first_frame_path).convert('RGBA')
    w, h = transparent_background.size
    transparent_layer = np.zeros((h, w, 4))
    for track in tracking_points.value:
        if len(track) > 1:
            for i in range(len(track)-1):
                start_point = track[i]
                end_point = track[i+1]
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(track)-2:
                    cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), color, 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, tuple(start_point), tuple(end_point), color, 2,)
        else:
            cv2.circle(transparent_layer, tuple(track[0]), 5, color, -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    return {tracking_points_var: tracking_points, input_image: trajectory_map}
    

def delete_last_step(tracking_points, first_frame_path, drag_mode):
    if drag_mode=='object':
        color = (255, 0, 0, 255)
    elif drag_mode=='camera':
        color = (0, 0, 255, 255)
    tracking_points.value[-1].pop()
    transparent_background = Image.open(first_frame_path).convert('RGBA')
    w, h = transparent_background.size
    transparent_layer = np.zeros((h, w, 4))
    for track in tracking_points.value:
        if len(track) > 1:
            for i in range(len(track)-1):
                start_point = track[i]
                end_point = track[i+1]
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(track)-2:
                    cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), color, 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, tuple(start_point), tuple(end_point), color, 2,)
        else:
            cv2.circle(transparent_layer, tuple(track[0]), 5,color, -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    return {tracking_points_var: tracking_points, input_image: trajectory_map}


if __name__=="__main__":
    block = gr.Blocks(
            theme=gr.themes.Soft(
                radius_size=gr.themes.sizes.radius_none,
                text_size=gr.themes.sizes.text_md
            )
            ).queue()
    with block as demo:
        with gr.Row():
            with gr.Column():
                gr.HTML(head)

        gr.Markdown(descriptions)

        with gr.Accordion(label="🛠️ Instructions:", open=True, elem_id="accordion"):
            with gr.Row(equal_height=True):
                gr.Markdown(instructions)      


        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device = torch.device("cuda")
        unet_path = 'models/unet.ckpt'
        image_controlnet_path = 'models/image_controlnet.ckpt'
        flow_controlnet_path = 'models/flow_controlnet.ckpt'
        ImageConductor_net = ImageConductor(device=device, 
                                            unet_path=unet_path, 
                                            image_controlnet_path=image_controlnet_path, 
                                            flow_controlnet_path=flow_controlnet_path, 
                                            height=256,
                                            width=384,
                                            model_length=16
                                            )
        first_frame_path_var = gr.State(value=None)
        tracking_points_var = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                image_upload_button = gr.UploadButton(label="Upload Image",file_types=["image"])
                add_drag_button = gr.Button(value="Add Drag")
                reset_button = gr.Button(value="Reset")
                delete_last_drag_button = gr.Button(value="Delete last drag")
                delete_last_step_button = gr.Button(value="Delete last step")
                
                

            with gr.Column(scale=7):
                with gr.Row():
                    with gr.Column(scale=6):
                        input_image = gr.Image(label="Input Image",
                                            interactive=True,
                                            height=300,
                                            width=384,)
                    with gr.Column(scale=6):
                        output_image = gr.Image(label="Motion Path",
                                                interactive=False,
                                                height=256,
                                                width=384,)
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(value="a wonderful elf.", label="Prompt (highly-recommended)", interactive=True, visible=True)
                negative_prompt = gr.Text(
                            label="Negative Prompt",
                            max_lines=5,
                            placeholder="Please input your negative prompt",
                            value='worst quality, low quality, letterboxed',lines=1
                        )
                drag_mode = gr.Radio(['camera', 'object'], label='Drag mode: ', value='object', scale=2)
                run_button = gr.Button(value="Run")

                with gr.Accordion("More input params", open=False, elem_id="accordion1"):
                    with gr.Group():
                        seed = gr.Textbox(
                            label="Seed: ",  value=561793204,
                        )
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                    
                    with gr.Group():
                        with gr.Row():
                            guidance_scale = gr.Slider(
                                label="Guidance scale",
                                minimum=1,
                                maximum=12,
                                step=0.1,
                                value=8.5,
                            )
                            num_inference_steps = gr.Slider(
                                label="Number of inference steps",
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=25,
                            )
                            
                    with gr.Group():
                        personalized = gr.Dropdown(label="Personalized", choices=["", 'HelloObject', 'TUSUN'], value="")
                        examples_type = gr.Textbox(label="Examples Type (Ignore) ",  value="", visible=False)

            with gr.Column(scale=7):
                output_video = gr.Video(
                                        label="Output Video", 
                                        width=384, 
                                        height=256)
                
                
        with gr.Row():
   
            example = gr.Examples(
                label="Input Example",
                examples=image_examples,
                inputs=[input_image, prompt, drag_mode, seed, personalized, examples_type],
                cache_examples=False,
            )
            
            
        with gr.Row():
            gr.Markdown(citation)

        
        image_upload_button.upload(preprocess_image, image_upload_button, [input_image, first_frame_path_var, tracking_points_var, personalized])

        add_drag_button.click(add_drag, [tracking_points_var], tracking_points_var)

        delete_last_drag_button.click(delete_last_drag, [tracking_points_var, first_frame_path_var, drag_mode], [tracking_points_var, input_image])

        delete_last_step_button.click(delete_last_step, [tracking_points_var, first_frame_path_var, drag_mode], [tracking_points_var, input_image])

        reset_button.click(reset_states, [first_frame_path_var, tracking_points_var], [input_image, first_frame_path_var, tracking_points_var])

        input_image.select(add_tracking_points, [tracking_points_var, first_frame_path_var, drag_mode], [tracking_points_var, input_image])

        run_button.click(ImageConductor_net.run, [first_frame_path_var, tracking_points_var, prompt, drag_mode, 
                                                negative_prompt, seed, randomize_seed, guidance_scale, num_inference_steps, personalized, examples_type], 
                                                [output_image, output_video])

        demo.launch(server_name="0.0.0.0", debug=True, server_port=12345)
