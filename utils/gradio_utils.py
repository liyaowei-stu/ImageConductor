# -*- coding:utf-8 -*-
import os
import sys
import shutil
from tqdm import tqdm
import yaml

import random
import importlib
from PIL import Image
import imageio

import numpy as np
import cv2
import torch
from torchvision import utils

from scipy.interpolate import PchipInterpolator

def split_filename(filename):
    absname = os.path.abspath(filename)
    dirname, basename = os.path.split(absname)
    split_tmp = basename.rsplit('.', maxsplit=1)
    if len(split_tmp) == 2:
        rootname, extname = split_tmp
    elif len(split_tmp) == 1:
        rootname = split_tmp[0]
        extname = None
    else:
        raise ValueError("programming error!")
    return dirname, rootname, extname


def data2file(data, filename, type=None, override=False, printable=False, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_did_not_save_flag = True
    if type:
        extname = type
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    if not os.path.exists(filename) or override:
        if extname in ['jpg', 'png', 'jpeg']:
            utils.save_image(data, filename, **kwargs)
        elif extname == 'gif':
            imageio.mimsave(filename, data, format='GIF', duration=kwargs.get('duration'), loop=0)
        elif extname == 'txt':
            if kwargs is None:
                kwargs = {}
            max_step = kwargs.get('max_step')
            if max_step is None:
                max_step = np.Infinity

            with open(filename, 'w', encoding='utf-8') as f:
                for i, e in enumerate(data):
                    if i < max_step:
                        f.write(str(e) + '\n')
                    else:
                        break
        else:
            raise ValueError('Do not support this type')
        if printable: print('Saved data to %s' % os.path.abspath(filename))
    else:
        if print_did_not_save_flag: print(
            'Did not save data to %s because file exists and override is False' % os.path.abspath(
                filename))


def file2data(filename, type=None, printable=True, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_load_flag = True
    if type:
        extname = type
    
    if extname in ['pth', 'ckpt', 'bin']:
        data = torch.load(filename, map_location=kwargs.get('map_location'))
        if "state_dict" in data.keys():
            data = data["state_dict"]
        data = {k.replace("_forward_module.", ""):v for k,v in data.items()}
    elif extname == 'txt':
        top = kwargs.get('top', None)
        with open(filename, encoding='utf-8') as f:
            if top:
                data = [f.readline() for _ in range(top)]
            else:
                data = [e for e in f.read().split('\n') if e]
    elif extname == 'yaml':
        with open(filename, 'r') as f:
            data = yaml.load(f)
        
    else:
        raise ValueError('type can only support h5, npy, json, txt')
    if printable:
        if print_load_flag:
            print('Loaded data from %s' % os.path.abspath(filename))
    return data


def ensure_dirname(dirname, override=False):
    if os.path.exists(dirname) and override:
        print('Removing dirname: %s' % os.path.abspath(dirname))
        try:
            shutil.rmtree(dirname)
        except OSError as e:
            raise ValueError('Failed to delete %s because %s' % (dirname, e))

    if not os.path.exists(dirname):
        print('Making dirname: %s' % os.path.abspath(dirname))
        os.makedirs(dirname, exist_ok=True)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def adaptively_load_state_dict(target, state_dict):
    target_dict = target.state_dict()

    try:
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict and v.size() == target_dict[k].size()}
        # unmatch_dict = {k: v for k, v in state_dict.items() if k not in target_dict or v.size() != target_dict[k].size()}
    except Exception as e:
        print('load error %s', e)
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict}

    if 'param_groups' in common_dict and common_dict['param_groups'][0]['params'] != \
            target.state_dict()['param_groups'][0]['params']:
        print('Detected mismatch params, auto adapte state_dict to current')
        common_dict['param_groups'][0]['params'] = target.state_dict()['param_groups'][0]['params']
    target_dict.update(common_dict)
    target.load_state_dict(target_dict)

    missing_keys = [k for k in target_dict.keys() if k not in common_dict]
    unexpected_keys = [k for k in state_dict.keys() if k not in common_dict]

    if len(unexpected_keys) != 0:
        print(
            f"Some weights of state_dict were not used in target: {unexpected_keys}"
        )
    if len(missing_keys) != 0:
        print(
            f"Some weights of state_dict are missing used in target {missing_keys}"
        )
    if len(unexpected_keys) == 0 and len(missing_keys) == 0:
        print("Strictly Loaded state_dict.")


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def image2pil(filename):
    return Image.open(filename)


def image2arr(filename):
    pil = image2pil(filename)
    return pil2arr(pil)


def pil2arr(pil):
    if isinstance(pil, list):
        arr = np.array(
            [np.array(e.convert('RGB').getdata(), dtype=np.uint8).reshape(e.size[1], e.size[0], 3) for e in pil])
    else:
        arr = np.array(pil)
    return arr


def arr2pil(arr):
    if arr.ndim == 3:
        return Image.fromarray(arr.astype('uint8'), 'RGB')
    elif arr.ndim == 4:
        return [Image.fromarray(e.astype('uint8'), 'RGB') for e in list(arr)]
    else:
        raise ValueError('arr must has ndim of 3 or 4, but got %s' % arr.ndim)


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

def visualize_drag(background_image_path, splited_tracks, width, height, model_length):
    background_image = Image.open(background_image_path).convert('RGBA')
    background_image = background_image.resize((width, height))
    w, h = background_image.size
    transparent_background = np.array(background_image)
    transparent_background[:, :, -1] = 128
    transparent_background = Image.fromarray(transparent_background)

    # Create a transparent layer with the same size as the background image
    transparent_layer = np.zeros((h, w, 4))
    for splited_track in splited_tracks:
        if len(splited_track) > 1:
            splited_track = interpolate_trajectory(splited_track, model_length)
            splited_track = splited_track[:model_length]
            for i in range(len(splited_track)-1):
                start_point = (int(splited_track[i][0]), int(splited_track[i][1]))
                end_point = (int(splited_track[i+1][0]), int(splited_track[i+1][1]))
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(splited_track)-2:
                    cv2.arrowedLine(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2)
        else:
            cv2.circle(transparent_layer, (int(splited_track[0][0]), int(splited_track[0][1])), 5, (255, 0,0, 192), -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    return trajectory_map, transparent_layer