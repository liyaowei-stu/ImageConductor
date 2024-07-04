import os
import imageio
import importlib

from omegaconf import OmegaConf
from typing import Union
from safetensors import safe_open
from tqdm import tqdm


import numpy as np
import torch
import torchvision
import torch.distributed as dist

from scipy.interpolate import PchipInterpolator
from einops import rearrange

from utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from utils.convert_lora_safetensor_to_diffusers import convert_lora, load_diffusers_lora
from modules.flow_controlnet import FlowControlNetModel
from modules.image_controlnet import ImageControlNetModel

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=2, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps, loop=0)


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


def load_weights(
    animation_pipeline,
    # motion module
    motion_module_path         = "",
    motion_module_lora_configs = [],
    # domain adapter
    adapter_lora_path          = "",
    adapter_lora_scale         = 1.0,
    # image layers
    dreambooth_model_path      = "",
    lora_model_path            = "",
    lora_alpha                 = 0.8,
):
    # motion module
    unet_state_dict = {}
    if motion_module_path != "":
        print(f"load motion module from {motion_module_path}")
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
        unet_state_dict.update({name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})
        unet_state_dict.pop("animatediff_config", "")
    
        missing, unexpected = animation_pipeline.unet.load_state_dict(unet_state_dict, strict=False)
        assert len(unexpected) == 0
        del unet_state_dict

    # base model
    if dreambooth_model_path != "":
        print(f"load dreambooth model from {dreambooth_model_path}")
        if dreambooth_model_path.endswith(".safetensors"):
            dreambooth_state_dict = {}
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
        elif dreambooth_model_path.endswith(".ckpt"):
            dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")
            
        # 1. vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, animation_pipeline.vae.config)
        for key in list(converted_vae_checkpoint.keys()):
            if 'mid_block' in key:
                if 'key' in key:
                    new_key = key.replace('key', 'to_k')
                elif 'query' in key:
                    new_key = key.replace('query', 'to_q')
                elif 'value' in key:
                    new_key = key.replace('value', 'to_v')
                elif 'proj_attn' in key:
                    new_key = key.replace('proj_attn', 'to_out.0')
                else: new_key=False
                if new_key:
                    converted_vae_checkpoint[new_key] = converted_vae_checkpoint[key]
                    del converted_vae_checkpoint[key]
        m, u = animation_pipeline.vae.load_state_dict(converted_vae_checkpoint, strict=False)
        print(f"dreambooth vae: {u}")
        # 2. unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, animation_pipeline.unet.config)
        m,u = animation_pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
        # 3. text_model
        animation_pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
        del dreambooth_state_dict
        
    # lora layers
    if lora_model_path != "":
        print(f"load lora model from {lora_model_path}")
        assert lora_model_path.endswith(".safetensors")
        lora_state_dict = {}
        with safe_open(lora_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                lora_state_dict[key] = f.get_tensor(key)
                
        animation_pipeline = convert_lora(animation_pipeline, lora_state_dict, alpha=lora_alpha)
        del lora_state_dict

    # domain adapter lora
    if adapter_lora_path != "":
        print(f"load domain lora from {adapter_lora_path}")
        domain_lora_state_dict = torch.load(adapter_lora_path, map_location="cpu")
        domain_lora_state_dict = domain_lora_state_dict["state_dict"] if "state_dict" in domain_lora_state_dict else domain_lora_state_dict
        domain_lora_state_dict.pop("animatediff_config", "")

        animation_pipeline = load_diffusers_lora(animation_pipeline, domain_lora_state_dict, alpha=adapter_lora_scale)

    # motion module lora
    for motion_module_lora_config in motion_module_lora_configs:
        path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]
        print(f"load motion LoRA from {path}")
        motion_lora_state_dict = torch.load(path, map_location="cpu")
        motion_lora_state_dict = motion_lora_state_dict["state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict
        motion_lora_state_dict.pop("animatediff_config", "")

        animation_pipeline = load_diffusers_lora(animation_pipeline, motion_lora_state_dict, alpha)

    return animation_pipeline


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_checkpoint(model_file, model):

    if not os.path.isfile(model_file):
        raise RuntimeError(f"{model_file} does not exist")
        
    state_dict = torch.load(model_file, map_location="cpu")
    global_step = state_dict['global_step'] if "global_step" in state_dict else 0
    new_state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    new_state_dict = {k.replace('module.', '') : v for k, v in new_state_dict.items()}
    m, u = model.load_state_dict(new_state_dict, strict=False)
    return model, global_step, m, u, new_state_dict


def load_model(model, model_path):
    if model_path != "":
        print(f"init model from checkpoint: {model_path}")
        model_ckpt = torch.load(model_path, map_location="cuda")
        if "global_step" in model_ckpt: print(f"global_step: {model_ckpt['global_step']}")
        state_dict = model_ckpt["state_dict"] if "state_dict" in model_ckpt else model_ckpt
        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0


def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points


def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    """Generate a bivariate isotropic or anisotropic Gaussian kernel.
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool):
    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.
    Args:
        kernel_size (int):
    Returns:
        xy (ndarray): with the shape (kernel_size, kernel_size, 2)
        xx (ndarray): with the shape (kernel_size, kernel_size)
        yy (ndarray): with the shape (kernel_size, kernel_size)
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size,
                                                                           1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def pdf2(sigma_matrix, grid):
    """Calculate PDF of the bivariate Gaussian distribution.
    Args:
        sigma_matrix (ndarray): with the shape (2, 2)
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.
    Returns:
        kernel (ndarrray): un-normalized kernel.
    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel


def sigma_matrix2(sig_x, sig_y, theta):
    """Calculate the rotated sigma matrix (two dimensional matrix).
    Args:
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
    Returns:
        ndarray: Rotated sigma matrix.
    """
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))


def create_image_controlnet(controlnet_config, unet, controlnet_path=""):
    # load controlnet model
    controlnet = None
        
    unet.config.num_attention_heads = 8
    unet.config.projection_class_embeddings_input_dim = None

    controlnet_config = OmegaConf.load(controlnet_config)
    controlnet = ImageControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

    if controlnet_path != "":
        print(f"loading controlnet checkpoint from {controlnet_path} ...")
        controlnet_state_dict = torch.load(controlnet_path, map_location="cuda")
        if "global_step" in controlnet_state_dict: print(f"global_step: {controlnet_state_dict['global_step']}")
        controlnet_state_dict = controlnet_state_dict["state_dict"] if "state_dict" in controlnet_state_dict else controlnet_state_dict
        controlnet_state_dict.pop("animatediff_config", "")
        controlnet.load_state_dict(controlnet_state_dict)
    
    return controlnet
       

def create_flow_controlnet(controlnet_config, unet, controlnet_path=""):
    # load controlnet model
    controlnet = None
        
    unet.config.num_attention_heads = 8
    unet.config.projection_class_embeddings_input_dim = None

    controlnet_config = OmegaConf.load(controlnet_config)
    controlnet = FlowControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

    if controlnet_path != "":
        print(f"loading controlnet checkpoint from {controlnet_path} ...")
        controlnet_state_dict = torch.load(controlnet_path, map_location="cuda")
        controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
        controlnet_state_dict.pop("animatediff_config", "")
        controlnet.load_state_dict(controlnet_state_dict)
    
    return controlnet
