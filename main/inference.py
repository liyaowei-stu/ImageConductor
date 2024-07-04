import argparse
import datetime
import inspect
import os, json
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image


import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from einops import rearrange, repeat
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available


from pipelines.pipeline_imagecoductor import ImageConductorPipeline
from modules.unet import UNet3DConditionFlowModel
from utils.visualizer import Visualizer, vis_flow_to_video
from utils.utils import create_image_controlnet, create_flow_controlnet, bivariate_Gaussian, save_videos_grid, load_weights, interpolate_trajectory, load_model
from utils.lora_utils import add_LoRA_to_controlnet



def view_trainable_param_name(model):
    trainable_name_lists = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            trainable_name_lists.append(name)
    return trainable_name_lists


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


@torch.no_grad()
def main(args):

    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)

    config  = OmegaConf.load(args.config)
    samples = []
    lora_rank = args.lora_rank

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    blur_kernel = bivariate_Gaussian(kernel_size=99, sig_x=10, sig_y=10, theta=0, grid=None, isotropic=True)

    # create validation pipeline
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)

        
        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionFlowModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

        ### >>> Initialize image controlnet >>> ###
        if model_config.image_controlnet_config is not None:
            image_controlnet = create_image_controlnet(model_config.image_controlnet_config, unet)
        
        ### >>> Initialize flow controlnet >>> ###
        if model_config.flow_controlnet_config is not None:
            flow_controlnet = create_flow_controlnet(model_config.flow_controlnet_config, unet)
            add_LoRA_to_controlnet(lora_rank, flow_controlnet)

        
        # Load pretrained unet weights
        unet_path = model_config.get("unet_path", "")
        load_model(unet, unet_path)
        
        # Load pretrained image controlnet weights
        image_controlnet_path = model_config.get("image_controlnet_path", "")
        load_model(image_controlnet, image_controlnet_path)

        # Load pretrained flow controlnet weights
        flow_controlnet_path = model_config.get("flow_controlnet_path", "")
        load_model(flow_controlnet, flow_controlnet_path)

        # load image condition
        controlnet_images = None
        if model_config.get("controlnet_images", "") != "":
            assert model_config.get("controlnet_images", "") != ""            
        
            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str): image_paths = [image_paths]

            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0), 
                    ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else: image_norm = lambda x: x
                
            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/sample{model_idx}_{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")
            
            num_controlnet_images = controlnet_images.shape[2]
            controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
            controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
            controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # load traj condition
        controlnet_flows = None
        if model_config.get("controlnet_trajs", "") != "":
            import ipdb; ipdb.set_trace()

            track_ponints_path= model_config.controlnet_trajs
            with open(track_ponints_path, 'r') as f:
                track_ponints = json.load(f)
            controlnet_flows = points_to_flows(track_ponints, model_config.L, model_config.H, model_config.W)

          
            for i in range(0, model_config.L-1):
                controlnet_flows[i] = cv2.filter2D(controlnet_flows[i], -1, blur_kernel)
            
            controlnet_flows = np.concatenate([np.zeros_like(controlnet_flows[0])[np.newaxis, ...], controlnet_flows], axis=0)  # pad the first frame with zero flow
            os.makedirs(os.path.join(savedir, "control_flows"), exist_ok=True)
            trajs_video = vis_flow_to_video(controlnet_flows, num_frames=model_config.L) # T-1 x H x W x 3
            torchvision.io.write_video(f'{savedir}/control_flows/sample{model_idx}_train_flow.mp4', trajs_video, fps=8, video_codec='h264', options={'crf': '10'})


            controlnet_flows = torch.from_numpy(controlnet_flows)[None].to(controlnet_images)[:, :model_config.L, ...]
            controlnet_flows =  rearrange(controlnet_flows, "b f h w c-> b c f h w")


        unet.to(device)
        image_controlnet.to(device)
        flow_controlnet.to(device)
        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()

        pipeline = ImageConductorPipeline(
            unet=unet,
            vae=vae, 
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            image_controlnet=image_controlnet,
            flow_controlnet=flow_controlnet,
        ).to(device)

        
        # load motion_module & domain adapter & dreambooth_model (optional), see Animatediff
        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = model_config.get("motion_module", ""),
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path          = model_config.get("adapter_lora_path", ""),
            adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ).to(device)
      
        

        prompts      = model_config.prompt
        n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
        
        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        
        config[model_idx].random_seed = []
        
        # traj_guidance_scales = [x / 2 for x in range(2, 7)]
        # traj_guidance_scales = [1.1]
        traj_guidance_scales = [1]
        vis = Visualizer(save_dir=f"{savedir}/sample", pad_value=0, linewidth=2, mode='cool', tracks_leave_trace=-1)


        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
            
            # manually set random seed for reproduction
            if random_seed != -1: torch.manual_seed(random_seed)
            else: torch.seed()
            config[model_idx].random_seed.append(torch.initial_seed())
            
            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")
            assert model_config.control_mode in ["object", "camera"], "control_mode in [object, camera]"
            sample = pipeline(
                prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = model_config.steps,
                guidance_scale      = model_config.guidance_scale,
                width               = model_config.W,
                height              = model_config.H,
                video_length        = model_config.L,
                controlnet_images = controlnet_images,
                controlnet_image_index = model_config.get("controlnet_image_indexs", [0]),
                controlnet_flows  = controlnet_flows,
                control_mode = model_config.control_mode,
                eval_mode = True,
            ).videos
            samples.append(sample)
            prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_videos_grid(sample, f"{savedir}/sample/sample-{sample_idx}-{prompt}.gif")

            if model_config.control_mode == "object":
                track_ponints_new = []
                for point in track_ponints:
                    splited_track = interpolate_trajectory(point, 16)
                    track_ponints_new.append(splited_track)
                points_track_vis = np.array(track_ponints_new).transpose(1, 0, 2)
                vis_video_obj= (sample[0] * 255).numpy().astype(np.uint8).transpose(1, 0, 2, 3)
                vis.visualize(torch.from_numpy(vis_video_obj[None]), torch.from_numpy(points_track_vis[None]), filename=f"track-traj-{sample_idx}-{prompt}", query_frame=0)

            print(f"save to {savedir}/sample/{prompt}.gif")
            sample_idx += 1

        samples = torch.concat(samples)
        save_videos_grid(samples, f"{savedir}/{model_idx}-sample.gif", n_rows=4)
        samples = []

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="models/stable-diffusion-v1-5",)
    parser.add_argument("--inference-config",      type=str, default="configs/inference/inference.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=384)
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--without-xformers", action="store_true")

    args = parser.parse_args()
    main(args)
