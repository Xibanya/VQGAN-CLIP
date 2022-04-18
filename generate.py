import warnings
# Supress warnings
warnings.filterwarnings('ignore')
import argparse
import gc
import math
import os
import re
import sys
# pip install taming-transformers doesn't work with Gumbel, but does not yet work with coco etc
# appending the path does work with Gumbel, but gives ModuleNotFoundError: No module named 'transformers' for coco etc
sys.path.append('taming-transformers')
from pathlib import Path
from subprocess import Popen, PIPE
from urllib.request import urlopen

import imageio
import kornia.augmentation as K
import numpy as np
import torch
import yaml
from PIL import ImageFile, Image, PngImagePlugin, ImageChops
from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
from torch import nn, optim
from torch.cuda import get_device_properties
from torch.nn import functional as F
from torch_optimizer import DiffGrad, AdamP, RAdam
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from rudalle import get_realesrgan
from rudalle.pipelines import super_resolution
from CLIP import clip

gc.collect()
torch.cuda.empty_cache()
CONFIG_PATH = 'config'
vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')
vq_parser.add_argument("-b", "--base", type=str, help="Config Path", default=CONFIG_PATH, dest='config_path')
args = vq_parser.parse_args()

CONFIG_DIRECTORY = 'Config'

VIDEO_CONFIG = 'video_config.yaml'
AUGMENT_CONFIG = 'augment_config.yaml'
with open(f"{CONFIG_DIRECTORY}/{args.config_path}.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
with open(f"{CONFIG_DIRECTORY}/{VIDEO_CONFIG}", "r") as f:
    vfg = yaml.load(f, Loader=yaml.FullLoader)
with open(f"{CONFIG_DIRECTORY}/{AUGMENT_CONFIG}", "r") as f:
    afg = yaml.load(f, Loader=yaml.FullLoader)

torch.backends.cudnn.benchmark = cfg['cudnn_benchmark']
# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation
# torch.use_deterministic_algorithms(True)

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"max_split_size_mb:{cfg['max_split_size_mb']}"

# Check for GPU and reduce the default image size if low VRAM
default_image_size = 512  # >8GB VRAM
if not torch.cuda.is_available():
    default_image_size = 256  # no GPU found
elif get_device_properties(0).total_memory <= 2 ** 33:  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
    default_image_size = 318  # <8GB VRAM

DEFAULT_CLIP_MODEL = 'ViT-B/32'
DEFAULT_PROMPT = '90s anime aesthetic'
DEFAULT_AUGMENTS = [['Af', 'Pe', 'Ji', 'Er']]


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_cyan(msg):
    print(f"{Colors.OKCYAN}{msg}{Colors.ENDC}")


def print_blue(msg):
    print(f"{Colors.OKBLUE}{msg}{Colors.ENDC}")


def print_green(msg):
    print(f"{Colors.OKGREEN}{msg}{Colors.ENDC}")


def print_warn(msg):
    print(f"{Colors.WARNING}{msg}{Colors.ENDC}")


def quantize_image(loaded_image, vq_model):
    loaded_image = loaded_image.convert('RGB')
    loaded_image = loaded_image.resize((sideX, sideY), Image.LANCZOS)
    loaded_tensor = TF.to_tensor(loaded_image)
    image_quant, *_ = vq_model.encode(loaded_tensor.to(device).unsqueeze(0) * 2 - 1)
    return image_quant


def has_alpha(image_to_check):
    return image_to_check.mode in ('RGBA', 'LA') or \
           image_to_check.mode == 'P' and 'transparency' in image_to_check.info


def quant_image_from_path(image_path: str, vq_model):
    if 'http' in image_path:
        loaded_image = Image.open(urlopen(image_path))
    else:
        loaded_image = nine_crop(Image.open(image_path), cfg['nine_type'])
    return quantize_image(loaded_image, vq_model)


def get_arg_label():
    label = ''
    if not afg['no_augments'] and afg['augments'] is not None:
        label += 'aug'
        for aug in afg['augments']:
            label += '.' + aug
        if afg['sharpness']['use'] and afg['sharpness']['arg'] not in afg['augments']:
            label += afg['sharpness']['arg']
        if afg['jitter']['use'] and afg['jitter']['arg'] not in afg['augments']:
            label += afg['jitter']['arg']
        if afg['erasing']['use'] and afg['erasing']['arg'] not in afg['augments']:
            label += afg['erasing']['arg']
        if afg['gaussian_noise']['use'] and afg['gaussian_noise']['arg'] not in afg['augments']:
            label += afg['gaussian_noise']['arg']
        if afg['gaussian_blur']['use'] and afg['gaussian_blur']['arg'] not in afg['augments']:
            label += afg['gaussian_blur']['arg']
        label += '_'
    return label


def get_prompt_label():
    label = '' if cfg['init_image'] is None else cfg['init_image'].split('.')[0] + '_'
    label += '' if cfg['prompts'] is None else \
        cfg['prompts'].replace(":", ".").replace(" | ", "-").replace("|", "-") + '_'

    label += '' if cfg['image_prompts'] is None else \
        cfg['image_prompts'].replace(f"{cfg['input_dir']}/", '') \
            .replace("|", "-").replace(".jpg", "").replace(".png", "") \
            .replace(".jpeg", "").replace(":", ".").replace("/", "") \
        + '_'
    if cfg['nine_type'] is not None:
        if isinstance(cfg['nine_type'], str):
            index = 6 if 'S' in cfg['nine_type'] else 3 if 'N' not in cfg['nine_type'] else 0
            toAdd = 2 if 'E' in cfg['nine_type'] else 1 if 'W' not in cfg['nine_type'] else 0
            index = index + toAdd
            label = f"{index:02d}_{label}"
        elif type(cfg['nine_type']) == int:
            index = cfg['nine_type']
            label = f"{index:02d}_{label}"
    return label


def get_config_label():
    label = '' if cfg['step_size'] == 0.1 else f"lr{cfg['step_size']}_"
    label += '' if cfg['max_iterations'] == 500 else f"i{cfg['max_iterations']}_"
    label += '' if cfg['seed'] is None else f"seed{cfg['seed']}_"
    label += '' if cfg['cutn'] == 0 else f"c{cfg['cutn']}_cp{cfg['cut_pow']}_"
    label += '' if cfg['cut_method'] == 'latest' else cfg['cut_method'] + '_'
    label += '' if cfg['optimiser'] == 'Adam' else f"{cfg['optimiser']}_"
    label += f"{cfg['weight_decay']}_" if cfg['optimiser'] == 'DiffGrad' else ''
    label += '_d_' if cfg['cudnn_determinism'] else ''
    label += cfg['clip_model'].replace('/', '') if \
        (cfg['clip_model'] != 'ViT-B/32' and cfg['clip_model'] is not None) else ''
    return label


def get_cut_label():
    label = ''
    return label


output_name = get_prompt_label() + get_arg_label() + get_config_label()
vqgan_type = cfg['vqgan_type']
if vqgan_type is not None:
    config_path = f"checkpoints/{vqgan_type}.yaml"
    if os.path.exists(config_path):
        vqgan_config = f"checkpoints/{vqgan_type}.yaml"
    else:
        # it's one of my custom models and I don't feel like copying
        # and pasting that same config file all the time
        vqgan_config = f"checkpoints/{cfg['default_config']}.yaml"
else:
    vqgan_type = cfg['default_model']
    vqgan_config = f"checkpoints/{cfg['default_model']}.yaml"

output_name += vqgan_type if vqgan_type != cfg['default_model'] else ''
out_dir = Path(cfg['output_dir']).resolve()
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
output = out_dir.joinpath(output_name + ".png")
in_dir = Path(cfg['input_dir']).resolve()
# prompts
prompts = cfg['prompts']
prompt_frequency = cfg['prompt_frequency']
noise_prompt_weights = cfg['noise_prompt_weights'] if cfg['noise_prompt_weights'] is not None else []
noise_prompt_seeds = cfg['noise_prompt_seeds'] if cfg['noise_prompt_seeds'] is not None else []
image_prompts = cfg['image_prompts'] if cfg['image_prompts'] is not None else []
init_image = str(in_dir.joinpath(cfg['init_image'])) if cfg['init_image'] and cfg['init_image'] != "None" else None

if not prompts and not image_prompts:
    prompts = DEFAULT_PROMPT

if cfg['cudnn_determinism']:
    torch.backends.cudnn.deterministic = True

augments = afg['augments'] if afg['augments'] else []
if not augments and not afg['no_augments']:
    augments = DEFAULT_AUGMENTS

# Split text prompts using the pipe character (weights are split later)
if prompts:
    # For stories, there will be many phrases
    story_phrases = [phrase.strip() for phrase in prompts.split("^")]

    # Make a list of all phrases
    all_phrases = []
    for phrase in story_phrases:
        all_phrases.append(phrase.split("|"))

    # First phrase
    prompts = all_phrases[0]

# Split target images using the pipe character (weights are split later)
if image_prompts:
    image_prompts = image_prompts.split("|")
    image_prompts = [image.strip() for image in image_prompts]

if vfg['make_video'] and vfg['make_zoom_video']:
    print_warn("Warning: Make video and make zoom video are mutually exclusive.")
    vfg['make_video'] = False

# Make video steps directory
if vfg['make_video'] or vfg['make_zoom_video']:
    if not os.path.exists('steps'):
        os.mkdir('steps')

# Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
# NB. May not work for AMD cards?
if not cfg['cuda_device'] == 'cpu' and not torch.cuda.is_available():
    cfg['cuda_device'] = 'cpu'
    print_warn("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
    print_warn("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")

# If a video_style_dir has been, then create a list of all the images
if vfg['video_style_dir']:
    print("Locating video frames...")
    video_frame_list = []
    for entry in os.scandir(vfg['video_style_dir']):
        if (entry.path.endswith(".jpg")
            or entry.path.endswith(".png")) and entry.is_file():
            video_frame_list.append(entry.path)

    # Reset a few options - same filename, different directory
    if not os.path.exists('steps'):
        os.mkdir('steps')

    init_image = video_frame_list[0]
    filename = os.path.basename(init_image)
    cwd = os.getcwd()
    output = os.path.join(cwd, "steps", filename)
    num_video_frames = len(video_frame_list)  # for video styling


# Various functions and classes
def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


# For zoom video
def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


# NR: Testing with different intital images
def random_noise_image(w, h):
    random_image = Image.fromarray(np.random.randint(0, 255, (w, h, 3), dtype=np.dtype('uint8')))
    return random_image


# create initial gradient image
def gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)

    return result


def random_gradient_image(w, h):
    array = gradient_3d(w, h, (0, 0, np.random.randint(0, 255)),
                        (np.random.randint(1, 255), np.random.randint(2, 255), np.random.randint(3, 128)),
                        (True, False, False))
    random_image = Image.fromarray(np.uint8(array))
    return random_image


# Used in older MakeCutouts
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


# NR: Split prompts and weights
def split_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


def get_color_jitter():
    jfg = afg['jitter']
    return K.ColorJitter(
        brightness=jfg['brightness'],
        contrast=jfg['contrast'],
        saturation=jfg['saturation'],
        hue=jfg['hue'],
        p=jfg['p'])


def get_sharpness():
    return K.RandomSharpness(
        sharpness=afg['sharpness']['sharpness'],
        p=afg['sharpness']['p'])


def get_gaussian_noise():
    return K.RandomGaussianNoise(
        mean=afg['gaussian_noise']['mean'],
        std=afg['gaussian_noise']['std'],
        p=afg['gaussian_noise']['p'])


def get_motion_blur():
    mblr = afg['motion_blur']
    return K.RandomMotionBlur(
        kernel_size=mblr['kernel_size'],
        angle=mblr['angle'],
        direction=mblr['direction'],
        border_type=mblr['border_type'],
        resample=mblr['resample'],
        same_on_batch=mblr['same_on_batch'],
        p=mblr['p'],
        keepdim=mblr['keepdim']
    )


def get_gaussian_blur():
    gblr = afg['gaussian_blur']
    return K.RandomGaussianBlur(
        kernel_size=gblr['kernel_size'],
        sigma=gblr['sigma'],
        border_type=gblr['border_type'],
        same_on_batch=gblr['same_on_batch'],
        p=gblr['p']
    )


def get_erasing():
    efg = afg['erasing']
    return K.RandomErasing(
        scale=efg['scale'],
        ratio=efg['ratio'],
        same_on_batch=efg['same_on_batch'],
        p=efg['p']
    )


def get_affine():
    cm = cfg['cut_method']
    aff = afg['affine']
    return K.RandomAffine(
        degrees=aff['degrees'],
        translate=(0.1, 0.1),
        shear=aff['shear'],
        p=aff['p'],
        padding_mode='border' if cm == 'updatedpooling' else 'zeros',
        keepdim=True)


def get_updated_pooling_augments():
    augment_list = [
        get_color_jitter(),
        get_erasing(),
        get_affine(),
        K.RandomPerspective(distortion_scale=0.7, p=0.7)
    ]
    return augment_list


def get_augment_list():
    augment_list = []
    cm = cfg['cut_method']

    if afg['no_augments']:
        if cm == 'updatedpooling':
            augment_list.append(get_color_jitter())
            augment_list.append(get_erasing())
            augment_list.append(get_affine())
            augment_list.append(K.RandomPerspective(
                distortion_scale=afg['perspective']['distortion_scale'],
                p=afg['perspective']['p']))
        else:
            dummy = get_color_jitter()
            dummy.p = 0.0
            augment_list.append(dummy)
        return augment_list

    # Xib TODO: make this respect order again
    if afg['jitter']['use'] or afg['jitter']['arg'] in augments[0] \
            or cm == 'updatedpooling':
        augment_list.append(get_color_jitter())
    if (afg['sharpness']['use'] or afg['sharpness']['arg'] in augments[0]) \
            and cm not in afg['sharpness']['incompatible']:
        augment_list.append(get_sharpness())
    if afg['gaussian_noise']['use']:
        augment_list.append(get_gaussian_noise())
    if afg['motion_blur']['use']:
        augment_list.append(get_motion_blur())
    if afg['gaussian_blur']['use']:
        augment_list.append(get_gaussian_blur())
    if (afg['erasing']['use'] or afg['erasing']['arg'] in augments[0]) \
            or cm == 'updatedpooling':
        augment_list.append(get_erasing())
    if (afg['affine']['use'] or afg['affine']['arg'] in augments[0]) \
            or cm == 'updatedpooling':
        augment_list.append(get_affine())
    if (afg['perspective']['use'] or afg['perspective']['arg'] in augments[0]) \
            or cm == 'updatedpooling':
        augment_list.append(K.RandomPerspective(
            distortion_scale=afg['perspective']['distortion_scale'],
            p=afg['perspective']['p']))
    if afg['crop']['use'] or afg['crop']['arg'] in augments[0]:
        augment_list.append(K.RandomCrop(
            size=(cut_size, cut_size),
            pad_if_needed=afg['crop']['pad_if_needed'],
            padding_mode=afg['crop']['padding_mode'],
            p=afg['crop']['p']))
    if afg['elastic_transform']['use'] or afg['elastic_transform']['arg'] in augments[0]:
        augment_list.append(K.RandomElasticTransform(p=afg['elastic_transform']['p']))
    if afg['rotation']['use'] or afg['rotation']['arg'] in augments[0]:
        augment_list.append(K.RandomRotation(
            degrees=afg['rotation']['degrees'],
            p=afg['rotation']['p']))
    if afg['resized_crop']['use'] or afg['resized_crop']['arg'] in augments[0]:
        rc = afg['resized_crop']
        augment_list.append(K.RandomResizedCrop(
            size=(cut_size, cut_size),
            scale=rc['scale'],
            ratio=rc['ratio'],
            cropping_mode=rc['cropping_mode'],
            p=rc['p']))
    if afg['thin_plate_spline']['use'] or afg['thin_plate_spline']['arg'] in augments[0]:
        tps = afg['thin_plate_spline']
        augment_list.append(K.RandomThinPlateSpline(
            scale=tps['scale'], same_on_batch=tps['same_on_batch'], p=tps['p']))

    return augment_list


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow  # not used with pooling

        # Pick your own augments & their order
        augment_list = get_augment_list()
        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = afg['noise_fac']

        # Uncomment if you like seeing the list ;)
        # print_green(augment_list)

        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []

        for _ in range(self.cutn):
            # Use Pooling
            cutout = (self.av_pool(input) + self.max_pool(input)) / 2
            cutouts.append(cutout)

        batch = self.augs(torch.cat(cutouts, dim=0))

        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An updated version with Kornia augments and pooling (where my version started):
# xibnote: ai art machine calls this "cumin"
class MakeCutoutsPoolingUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow  # Not used with pooling

        augment_list = get_updated_pooling_augments()
        self.augs = nn.Sequential(*augment_list)

        self.noise_fac = afg['noise_fac']
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []

        for _ in range(self.cutn):
            cutout = (self.av_pool(input) + self.max_pool(input)) / 2
            cutouts.append(cutout)

        batch = self.augs(torch.cat(cutouts, dim=0))

        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An Nerdy updated version with selectable Kornia augments, but no pooling:
class MakeCutoutsNRUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = afg['noise_fac']

        # Pick your own augments & their order
        augment_list = get_augment_list()

        self.augs = nn.Sequential(*augment_list)

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An updated version with Kornia augments, but no pooling:
class MakeCutoutsUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            get_color_jitter(),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2, p=0.4), )
        self.noise_fac = afg['noise_fac']

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# This is the original version (No pooling)
class MakeCutoutsOrig(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


def load_vqgan_model(config_path, checkpoint_path):
    global gumbel
    gumbel = False
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


# Do it
device = torch.device(cfg['cuda_device'])
print_green(f"\nUsing device: {device}")
model = load_vqgan_model(vqgan_config, f"checkpoints/{vqgan_type}.ckpt").to(device)
jit = False  # True if float(torch.__version__[:3]) < 1.8 else False

clip_model = cfg['clip_model'] if cfg['clip_model'] is not None else DEFAULT_CLIP_MODEL
perceptor = clip.load(clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)
print_blue(f"CLIP Model: {clip_model}")
print_blue('Optimising using: ' + cfg['optimiser'])

# clock=deepcopy(perceptor.visual.positional_embedding.data)
# perceptor.visual.positional_embedding.data = clock/clock.max()
# perceptor.visual.positional_embedding.data=clamp_with_grad(clock,0,1)

cut_size = perceptor.visual.input_resolution
f = 2 ** (model.decoder.num_resolutions - 1)

# Cutout class options:
# 'latest','original','updated' or 'updatedpooling'
if cfg['cut_method'] == 'latest':
    make_cutouts = MakeCutouts(cut_size, cfg['cutn'], cut_pow=cfg['cut_pow'])
elif cfg['cut_method'] == 'original':
    make_cutouts = MakeCutoutsOrig(cut_size, cfg['cutn'], cut_pow=cfg['cut_pow'])
elif cfg['cut_method'] == 'updated':
    make_cutouts = MakeCutoutsUpdate(cut_size, cfg['cutn'], cut_pow=cfg['cut_pow'])
elif cfg['cut_method'] == 'nrupdated':
    make_cutouts = MakeCutoutsNRUpdate(cut_size, cfg['cutn'], cut_pow=cfg['cut_pow'])
else:
    make_cutouts = MakeCutoutsPoolingUpdate(cut_size, cfg['cutn'], cut_pow=cfg['cut_pow'])

toksX, toksY = cfg['size'][0] // f, cfg['size'][1] // f
sideX, sideY = toksX * f, toksY * f


def nine_crop(to_crop, nine_type):
    if nine_type is None:
        return to_crop
    width, height = to_crop.size
    mX = width / 4
    mY = height / 4
    w = width / 2
    h = height / 2

    if isinstance(nine_type, str):
        if 'N' in nine_type:
            top = 0
            bottom = h
        elif 'S' in nine_type:
            top = height - h
            bottom = height
        else:
            top = mY
            bottom = height - mY
        if 'W' in nine_type:
            left = 0
            right = w
        elif 'E' in nine_type:
            right = width
            left = width - w
        else:
            left = mX
            right = width - mX
    elif type(nine_type) == int:
        if nine_type < 3:
            top = 0
            bottom = h
        elif nine_type > 5:
            top = height - h
            bottom = height
        else:
            top = mY
            bottom = height - mY
        if nine_type % 3 == 0:
            left = 0
            right = w
        elif (nine_type + 1) % 3 == 0:
            right = width
            left = width - w
        else:
            left = mX
            right = width - mX
    else:
        print_warn(f"nine_type type {type(nine_type)} unexpected!")
        return to_crop

    print_green(f"Nine type: {nine_type}")
    cropped = to_crop.crop((left, top, right, bottom))
    realesrgan = get_realesrgan(cfg['realesrgan_model'], device='cuda')
    cropped = super_resolution([cropped], realesrgan)[0]
    return cropped.resize((cfg['size'][0], cfg['size'][1]))


if cfg['init_image'] is not None and cfg['ignore_alpha'] is False:
    start_img = Image.open(f"{cfg['input_dir']}/{cfg['init_image']}")
    start_img = nine_crop(start_img, str(cfg['nine_type']))
    init_image_has_alpha = has_alpha(start_img)
    im_a = resize_image(start_img.convert('RGBA').split()[-1], (sideX, sideY)) if init_image_has_alpha else None
else:
    init_image_has_alpha = False
    im_a = None

# Gumbel or not?
if gumbel:
    e_dim = 256
    n_toks = model.quantize.n_embed
    z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
else:
    e_dim = model.quantize.e_dim
    n_toks = model.quantize.n_e
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

if init_image:
    z = quant_image_from_path(init_image, model)
elif cfg['init_noise'] == 'pixels':
    img = random_noise_image(cfg['size'][0], cfg['size'][1])
    z = quantize_image(img, model)
elif cfg['init_noise'] == 'gradient':
    img = random_gradient_image(cfg['size'][0], cfg['size'][1])
    z = quantize_image(img, model)
else:
    one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
    # z = one_hot @ model.quantize.embedding.weight
    if gumbel:
        z = one_hot @ model.quantize.embed.weight
    else:
        z = one_hot @ model.quantize.embedding.weight

    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    # z = torch.rand_like(z)*2						# NR: check

z_orig = z.clone()
z.requires_grad_(True)

pMs = []
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

# From imagenet - Which is better?
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

# CLIP tokenize/encode   
if prompts:
    for prompt in prompts:
        txt, weight, stop = split_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

for prompt in image_prompts:
    path, weight, stop = split_prompt(prompt)
    if cfg['input_dir'] not in path:
        path = f"{cfg['input_dir']}/{path}"
    img = Image.open(path)
    pil_image = img.convert('RGB')
    img = resize_image(pil_image, (sideX, sideY))
    batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
    embed = perceptor.encode_image(normalize(batch)).float()
    pMs.append(Prompt(embed, weight, stop).to(device))

for seed, weight in zip(noise_prompt_seeds, noise_prompt_weights):
    gen = torch.Generator().manual_seed(seed)
    embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
    pMs.append(Prompt(embed, weight).to(device))


# Set the optimiser
def get_opt(opt_name, opt_lr):
    if opt_name == "Adam":
        opt = optim.Adam([z], lr=opt_lr)  # LR=0.1 (Default)
    elif opt_name == "AdamW":
        opt = optim.AdamW([z], lr=opt_lr)
    elif opt_name == "Adagrad":
        opt = optim.Adagrad([z], lr=opt_lr)
    elif opt_name == "Adamax":
        opt = optim.Adamax([z], lr=opt_lr)
    elif opt_name == "DiffGrad":
        opt = DiffGrad([z], lr=opt_lr, eps=1e-9, weight_decay=float(cfg['weight_decay']))  # NR: Playing for reasons
    elif opt_name == "AdamP":
        opt = AdamP([z], lr=opt_lr)
    elif opt_name == "RAdam":
        opt = RAdam([z], lr=opt_lr)
    elif opt_name == "RMSprop" or opt_name == "RMSProp":
        opt = optim.RMSprop([z], lr=opt_lr)
    else:
        print_warn("Unknown optimiser. Are choices broken?")
        opt = optim.Adam([z], lr=opt_lr)
    return opt


opt = get_opt(cfg['optimiser'], cfg['step_size'])

if prompts:
    print_cyan(f"\nUsing text prompts: {prompts}")
if image_prompts:
    print_cyan(f"Using image prompts: {image_prompts}")
if init_image:
    print_cyan('Using initial image: ' + init_image)
if noise_prompt_weights:
    print_cyan(f"Noise prompt weights: {noise_prompt_weights}")

if cfg['seed'] is None:
    seed = torch.seed()
else:
    seed = cfg['seed']
torch.manual_seed(seed)
print_cyan(f"Using seed: {seed}")


# Vector quantize
def synth(z):
    if gumbel:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
    else:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


# @torch.no_grad()
@torch.inference_mode()
def checkin(i, losses):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
    out = synth(z)
    info = PngImagePlugin.PngInfo()
    info.add_text('comment', f'{prompts}')
    info.add_text("Iterations", str(i))

    if i == cfg['max_iterations'] and init_image_has_alpha and im_a is not None:
        new_img = TF.to_pil_image(out[0].cpu())
        try:
            new_img.putalpha(im_a)
            if cfg['realesrgan']:
                images = [new_img]
                realesrgan = get_realesrgan(cfg['realesrgan_model'], device='cuda')
                images = super_resolution(images, realesrgan)
                new_img = images[0]
        except Exception:
            print_warn("exception putting alpha")
        new_img.save(output, pnginfo=info)
    else:
        new_img = TF.to_pil_image(out[0].cpu())
        if cfg['realesrgan'] and (i == cfg['max_iterations'] or cfg['upscale_always']):
            images = [new_img]
            realesrgan = get_realesrgan(cfg['realesrgan_model'], device='cuda')
            images = super_resolution(images, realesrgan)
            new_img = images[0]
        new_img.save(output, pnginfo=info)


def ascend_txt():
    global i
    out = synth(z)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

    result = []

    if cfg['init_weight']:
        # result.append(F.mse_loss(z, z_orig) * cfg['init_weight'] / 2)
        result.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1 / torch.tensor(i * 2 + 1)) * cfg['init_weight']) / 2)

    for prompt in pMs:
        result.append(prompt(iii))

    if vfg['make_video']:
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        imageio.imwrite('./steps/' + str(i) + '.png', np.array(img))

    return result  # return loss


def train(i):
    opt.zero_grad(set_to_none=True)
    lossAll = ascend_txt()

    if i % cfg['display_freq'] == 0:
        checkin(i, lossAll)

    loss = sum(lossAll)
    loss.backward()
    opt.step()

    # with torch.no_grad():
    with torch.inference_mode():
        z.copy_(z.maximum(z_min).minimum(z_max))


i = 0  # Iteration counter
j = 0  # Zoom video frame counter
p = 1  # Phrase counter
smoother = 0  # Smoother counter
this_video_frame = 0  # for video styling

# Do it
try:
    with tqdm() as pbar:
        while True:
            # Change generated image
            if vfg['make_zoom_video']:
                if i % vfg['zoom_frequency'] == 0:
                    out = synth(z)

                    # Save image
                    img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
                    img = np.transpose(img, (1, 2, 0))
                    imageio.imwrite('./steps/' + str(j) + '.png', np.array(img))

                    # Time to start zooming?                    
                    if vfg['zoom_start'] <= i:
                        # Convert z back into a Pil image                    
                        # pil_image = TF.to_pil_image(out[0].cpu())

                        # Convert NP to Pil image
                        pil_image = Image.fromarray(np.array(img).astype('uint8'), 'RGB')

                        # Zoom
                        if vfg['zoom_scale'] != 1:
                            pil_image_zoom = zoom_at(pil_image, sideX / 2, sideY / 2, vfg['zoom_scale'])
                        else:
                            pil_image_zoom = pil_image

                        # Shift - https://pillow.readthedocs.io/en/latest/reference/ImageChops.html
                        if vfg['zoom_shift_x'] or vfg['zoom_shift_y']:
                            # This one wraps the image
                            pil_image_zoom = ImageChops.offset(pil_image_zoom, vfg['zoom_shift_x'], vfg['zoom_shift_y'])

                        # Convert image back to a tensor again
                        pil_tensor = TF.to_tensor(pil_image_zoom)

                        # Re-encode
                        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                        z_orig = z.clone()
                        z.requires_grad_(True)

                        # Re-create optimiser
                        opt = get_opt(cfg['optimiser'], cfg['step_size'])

                    # Next
                    j += 1

            # Change text prompt
            if prompt_frequency is not None and prompt_frequency > 0:
                if i % prompt_frequency == 0 and i > 0:
                    # In case there aren't enough phrases, just loop
                    if p >= len(all_phrases):
                        p = 0

                    pMs = []
                    prompts = all_phrases[p]

                    # Show user we're changing prompt                                
                    print_green(prompts)

                    for prompt in prompts:
                        txt, weight, stop = split_prompt(prompt)
                        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                        pMs.append(Prompt(embed, weight, stop).to(device))
                    p += 1

            # Training time
            train(i)

            # Ready to stop yet?
            if i == cfg['max_iterations']:
                if not vfg['video_style_dir']:
                    # we're done
                    break
                else:
                    if this_video_frame == (num_video_frames - 1):
                        # we're done
                        make_styled_video = True
                        break
                    else:
                        # Next video frame
                        this_video_frame += 1

                        # Reset the iteration count
                        i = -1
                        pbar.reset()

                        # Load the next frame, reset a few options - same filename, different directory
                        init_image = video_frame_list[this_video_frame]
                        print("Next frame: ", init_image)

                        if cfg['seed'] is None:
                            seed = torch.seed()
                        else:
                            seed = cfg['seed']
                        torch.manual_seed(seed)
                        print_green("Seed: " + seed)

                        filename = os.path.basename(init_image)
                        output = os.path.join(cwd, "steps", filename)

                        # Load and resize image
                        img = Image.open(init_image)
                        pil_image = img.convert('RGB')
                        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
                        pil_tensor = TF.to_tensor(pil_image)

                        # Re-encode
                        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                        z_orig = z.clone()
                        z.requires_grad_(True)

                        # Re-create optimiser
                        opt = get_opt(cfg['optimiser'], cfg['step_size'])

            i += 1
            pbar.update()
except KeyboardInterrupt:
    pass

# All done :)

# Video generation
if vfg['make_video'] or vfg['make_zoom_video']:
    init_frame = 1  # Initial video frame
    if vfg['make_zoom_video']:
        last_frame = j
    else:
        last_frame = i  # This will raise an error if that number of frames does not exist.

    length = vfg['video_length']  # Desired time of the video in seconds

    min_fps = 10
    max_fps = 60

    total_frames = last_frame - init_frame

    frames = []
    tqdm.write('Generating video...')
    for i in range(init_frame, last_frame):
        temp = Image.open("./steps/" + str(i) + '.png')
        keep = temp.copy()
        frames.append(keep)
        temp.close()

    if vfg['output_video_fps'] > 9:
        # Hardware encoding and video frame interpolation
        print("Creating interpolated frames...")
        ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps={str(vfg['output_video_fps'])}'"
        output_file = re.compile('\.png$').sub('.mp4', output)
        try:
            p = Popen(['ffmpeg',
                       '-y',
                       '-f', 'image2pipe',
                       '-vcodec', 'png',
                       '-r', str(vfg['input_video_fps']),
                       '-i',
                       '-',
                       '-b:v', '10M',
                       '-vcodec', 'h264_nvenc',
                       '-pix_fmt', 'yuv420p',
                       '-strict', '-2',
                       '-filter:v', f'{ffmpeg_filter}',
                       '-metadata', f'comment={prompts}',
                       output_file], stdin=PIPE)
        except FileNotFoundError:
            print_warn("ffmpeg command failed - check your installation")
        for im in tqdm(frames):
            im.save(p.stdin, 'PNG')
        p.stdin.close()
        p.wait()
    else:
        # CPU
        fps = np.clip(total_frames / length, min_fps, max_fps)
        output_file = re.compile('\.png$').sub('.mp4', output)
        try:
            p = Popen(['ffmpeg',
                       '-y',
                       '-f', 'image2pipe',
                       '-vcodec', 'png',
                       '-r', str(fps),
                       '-i',
                       '-',
                       '-vcodec', 'libx264',
                       '-r', str(fps),
                       '-pix_fmt', 'yuv420p',
                       '-crf', '17',
                       '-preset', 'veryslow',
                       '-metadata', f'comment={prompts}',
                       output_file], stdin=PIPE)
        except FileNotFoundError:
            print_warn("ffmpeg command failed - check your installation")
        for im in tqdm(frames):
            im.save(p.stdin, 'PNG')
        p.stdin.close()
        p.wait()
