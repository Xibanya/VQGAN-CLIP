import os
import gc
import yaml
import torch
from torch import optim
from torch.cuda import get_device_properties
from torch_optimizer import DiffGrad, AdamP, RAdam
from pathlib import Path
from PIL import ImageFile
from .Colors import print_warn, print_blue, print_cyan


gc.collect()
torch.cuda.empty_cache()

CONFIG_DIRECTORY = 'Config/config.yaml'
DEFAULT_CLIP_MODEL = 'ViT-B/32'
DEFAULT_PROMPT = '90s anime aesthetic'

AUGMENT_CONFIG = 'augment_config.yaml'
with open(CONFIG_DIRECTORY, "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"max_split_size_mb:{cfg['max_split_size_mb']}"
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
# NB. May not work for AMD cards?
if not cfg['cuda_device'] == 'cpu' and not torch.cuda.is_available():
    cfg['cuda_device'] = 'cpu'
    print_warn("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
    print_warn("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")

# Check for GPU and reduce the default image size if low VRAM
default_image_size = 512  # >8GB VRAM
if not torch.cuda.is_available():
    default_image_size = 256  # no GPU found
elif get_device_properties(0).total_memory <= 2 ** 33:  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
    default_image_size = 318  # <8GB VRAM

torch.backends.cudnn.benchmark = cfg['cudnn_benchmark']
if cfg['cudnn_determinism']:
    torch.backends.cudnn.deterministic = True


def get_nine_prefix(nine_type):
    if isinstance(nine_type, str):
        index = 6 if 'S' in nine_type else 3 if 'N' not in nine_type else 0
        toAdd = 2 if 'E' in nine_type else 1 if 'W' not in nine_type else 0
        index = index + toAdd
        return f"{index}_"
    elif type(nine_type) == int:
        index = nine_type
        return f"{index}_"
    else:
        return ""


def get_prompt_label():
    label = '' if cfg['init_image'] is None else cfg['init_image'].split('.')[0] + '_'
    label += '' if cfg['prompts'] is None else \
        cfg['prompts'].replace(":", ".").replace(" | ", "-").replace("|", "-") + '_'

    label += '' if cfg['image_prompts'] is None else \
        cfg['image_prompts'].replace(f"{cfg['input_dir']}/", '') \
            .replace("|", "-").replace(".jpg", "").replace(".png", "") \
            .replace(".jpeg", "").replace(":", ".").replace("/", "") \
        + '_'
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


def get_output_path(nine_type):
    if nine_type is not None:
        output_name = get_nine_prefix(nine_type) + cfg['init_image'].split('.')[0]
    else:
        output_name = get_prompt_label() + get_config_label()
        vqgan_type = cfg['vqgan_type'] if cfg['vqgan_type'] is not None else cfg['default_model']
        output_name += vqgan_type if vqgan_type != cfg['default_model'] else ''
    out_dir = Path(cfg['output_dir']).resolve()
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    output = out_dir.joinpath(output_name + ".png")
    return output


def get_all_phrases():
    prompts = cfg['prompts']
    all_phrases = []
    if prompts:
        # For stories, there will be many phrases
        story_phrases = [phrase.strip() for phrase in prompts.split("^")]

        # Make a list of all phrases
        for phrase in story_phrases:
            all_phrases.append(phrase.split("|"))
    return all_phrases


def get_prompts(nine_type):
    # prompts
    prompts = cfg['prompts']
    if type(nine_type) == int and cfg['slice_prompts'][nine_type] is not None:
        if prompts is not None:
            prompts = prompts + "|"
        prompts = prompts + cfg['slice_prompts'][nine_type]

    prompt_frequency = cfg['prompt_frequency']
    if not prompts and not cfg['image_prompts']:
        prompts = DEFAULT_PROMPT
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
    if prompts:
        print_cyan(f"\nUsing text prompts: {prompts}")
    return prompts


def get_image_prompts():
    # Split target images using the pipe character (weights are split later)
    if cfg['image_prompts'] is not None:
        image_prompts = cfg['image_prompts'].split("|")
        image_prompts = [image.strip() for image in image_prompts]
        print_cyan(f"Using image prompts: {image_prompts}")
    else:
        image_prompts = []
    return image_prompts


def get_opt(opt_name, opt_lr, z):
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
    print_blue('Optimising using: ' + cfg['optimiser'])
    return opt


def get_perceptor(device, clip):
    clip_model = cfg['clip_model'] if cfg['clip_model'] is not None else DEFAULT_CLIP_MODEL
    print_blue(f"CLIP Model: {clip_model}")
    perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    return perceptor


def set_seed():
    if cfg['seed'] is None:
        seed = torch.seed()
    else:
        seed = cfg['seed']
    torch.manual_seed(seed)
    print_cyan(f"Using seed: {seed}")
