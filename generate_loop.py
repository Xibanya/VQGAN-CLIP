import warnings

# Supress warnings
warnings.filterwarnings('ignore')

import os
import sys
from urllib.request import urlopen

sys.path.append('taming-transformers')
from pathlib import Path

from PIL import PngImagePlugin
from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
from datetime import datetime
from rudalle import get_realesrgan
from rudalle.pipelines import super_resolution
from CLIP import clip
from XibGAN.Colors import print_green, print_cyan, print_warn
from XibGAN.Utils import *
from XibGAN.Config import (
    cfg, get_opt, get_perceptor, get_prompts, get_image_prompts, get_all_phrases, get_output_path,
    set_seed, get_output_dir
)
from XibGAN.Augments import get_cutout_function
from XibGAN.Composite import paste_on_base, paste_on_overlay


def get_start_image():
    in_dir = Path(cfg['input_dir']).resolve()
    init_image = str(in_dir.joinpath(cfg['init_image'])) if cfg['init_image'] and cfg['init_image'] != "None" else None
    return Image.open(init_image)


now = datetime.now().strftime("%dT%H-%M-%S")
filename = f"{cfg['init_image'].split('.')[0]}_Composite_{now}.png"
path = get_output_dir().joinpath(filename)


# Vector quantize
def synth(z, model):
    if gumbel:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
    else:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


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
    # each slice is half the size of the output res,
    # so will upscale x2 to avoid regular resize artifacts
    #realesrgan = get_realesrgan('x2', device='cuda')
    #cropped = super_resolution([cropped], realesrgan)[0]
    return cropped.resize((cfg['size'][0], cfg['size'][1]))


def quantize_image(loaded_image, vq_model, sideX, sideY, device):
    loaded_image = loaded_image.convert('RGB')
    loaded_image = loaded_image.resize((sideX, sideY), Image.LANCZOS)
    loaded_tensor = TF.to_tensor(loaded_image)
    image_quant, *_ = vq_model.encode(loaded_tensor.to(device).unsqueeze(0) * 2 - 1)
    return image_quant


def quant_image_from_path(image_path: str, vq_model, nine_type, sideX, sideY, device):
    if 'http' in image_path:
        loaded_image = Image.open(urlopen(image_path))
    else:
        loaded_image = nine_crop(Image.open(image_path), nine_type)
    return quantize_image(loaded_image, vq_model, sideX, sideY, device)


def put_alpha(new_img, nine_type):
    in_dir = Path(cfg['input_dir']).resolve()
    init_image = str(in_dir.joinpath(cfg['init_image'])) if cfg['init_image'] and cfg['init_image'] != "None" else None
    if init_image is not None and cfg['ignore_alpha'] is False:
        start_img = Image.open(init_image)
        start_img = nine_crop(start_img, str(nine_type))
        init_image_has_alpha = has_alpha(start_img)
        w, h = new_img.size
        im_a = resize_image(start_img.convert('RGBA').split()[-1], (w, h)) if init_image_has_alpha else None
    else:
        init_image_has_alpha = False
        im_a = None
    if init_image_has_alpha and im_a is not None:
        try:
            new_img.putalpha(im_a)
        except Exception:
            print_warn("exception putting alpha")
    return new_img


@torch.inference_mode()
def checkin(i, losses, model, z, nine_type):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
    out = synth(z, model)
    info = PngImagePlugin.PngInfo()
    info.add_text("Iterations", str(i))
    info.add("Clip", cfg['clip_model'])
    info.add("Method", cfg['cut_method'])
    new_img = TF.to_pil_image(out[0].cpu())
    if i == cfg['max_iterations'] and not cfg['ignore_alpha']:
        new_img = put_alpha(new_img, nine_type)
    if nine_type is None and cfg['realesrgan'] and \
            (i == cfg['max_iterations'] or cfg['upscale_always']):
        images = [new_img]
        realesrgan = get_realesrgan(cfg['realesrgan_model'], device='cuda')
        images = super_resolution(images, realesrgan)
        new_img = images[0]
    if nine_type is None or cfg['save_nine']:
        new_img.save(get_output_path(nine_type), pnginfo=info)

    if nine_type is not None:
        composite = Image.open(path)
        composite = paste_on_base(composite, new_img, nine_type, path)
        composite = paste_on_overlay(composite, new_img, nine_type, cfg['smooth'], path)
        return composite
    else:
        return new_img


def ascend_txt(i, z, perceptor, z_orig, make_cutouts, prompts, model):
    out = synth(z, model)
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
    result = []
    if cfg['init_weight']:
        result.append(
            F.mse_loss(z, torch.zeros_like(z_orig)) * ((1 / torch.tensor(i * 2 + 1)) * cfg['init_weight']) / 2)
    for prompt in prompts:
        result.append(prompt(iii))
    return result  # return loss


def train(i, opt, z, z_min, z_max, perceptor, z_orig, make_cutouts, prompts, model, nine_type, current_img):
    opt.zero_grad(set_to_none=True)
    lossAll = ascend_txt(i, z, perceptor, z_orig, make_cutouts, prompts, model)

    if i % cfg['display_freq'] == 0:
        current_img = checkin(i, lossAll, z=z, model=model, nine_type=nine_type)

    loss = sum(lossAll)
    loss.backward()
    opt.step()

    # with torch.no_grad():
    with torch.inference_mode():
        z.copy_(z.maximum(z_min).minimum(z_max))
    return current_img


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


def do_it(nine_type):
    device = torch.device(cfg['cuda_device'])
    print_green(f"\nUsing device: {device}")
    vqgan_type = cfg['vqgan_type'] if cfg['vqgan_type'] is not None else cfg['default_model']
    if cfg['vqgan_type'] is not None:
        config_path = f"checkpoints/{vqgan_type}.yaml"
        if os.path.exists(config_path):
            vqgan_config = f"checkpoints/{vqgan_type}.yaml"
        else:
            # it's one of my custom models and I don't feel like copying
            # and pasting that same config file all the time
            vqgan_config = f"checkpoints/{cfg['default_config']}.yaml"
    else:
        vqgan_config = f"checkpoints/{cfg['default_model']}.yaml"
    model = load_vqgan_model(vqgan_config, f"checkpoints/{vqgan_type}.ckpt").to(device)
    perceptor = get_perceptor(device, clip)
    cut_size = perceptor.visual.input_resolution
    f = 2 ** (model.decoder.num_resolutions - 1)
    make_cutouts = get_cutout_function(cfg, cut_size)

    toksX, toksY = cfg['size'][0] // f, cfg['size'][1] // f
    sideX, sideY = toksX * f, toksY * f

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

    in_dir = Path(cfg['input_dir']).resolve()
    init_image = str(in_dir.joinpath(cfg['init_image'])) if cfg['init_image'] and cfg['init_image'] != "None" else None

    if cfg['init_image']:
        if init_image:
            print_cyan('Using initial image: ' + init_image)
        z = quant_image_from_path(
            init_image, model, nine_type,
            sideX=sideX, sideY=sideY, device=device
        )
    elif cfg['init_noise'] == 'pixels':
        img = random_noise_image(cfg['size'][0], cfg['size'][1])
        z = quantize_image(img, model, sideX=sideX, sideY=sideY, device=device)
    elif cfg['init_noise'] == 'gradient':
        img = random_gradient_image(cfg['size'][0], cfg['size'][1])
        z = quantize_image(img, model, sideX=sideX, sideY=sideY, device=device)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        if gumbel:
            z = one_hot @ model.quantize.embed.weight
        else:
            z = one_hot @ model.quantize.embedding.weight

        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

    z_orig = z.clone()
    z.requires_grad_(True)
    opt = get_opt(cfg['optimiser'], cfg['step_size'], z)
    pMs = []

    prompts = get_prompts(nine_type)
    image_prompts = get_image_prompts()
    all_phrases = get_all_phrases()
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
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    noise_prompt_weights = cfg['noise_prompt_weights'] if cfg['noise_prompt_weights'] is not None else []
    if noise_prompt_weights:
        print_cyan(f"Noise prompt weights: {noise_prompt_weights}")
    noise_prompt_seeds = cfg['noise_prompt_seeds'] if cfg['noise_prompt_seeds'] is not None else []
    for seed, weight in zip(noise_prompt_seeds, noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

    set_seed()

    i = 0  # Iteration counter
    p = 1  # Phrase counter
    current_img = Image.open(init_image)

    # Do it
    try:
        with tqdm() as pbar:
            while True:
                # Change text prompt
                if cfg['prompt_frequency'] is not None and cfg['prompt_frequency'] > 0:
                    if i % cfg['prompt_frequency'] == 0 and i > 0:
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
                current_img = train(i, opt, z, z_min, z_max, perceptor, z_orig, make_cutouts, pMs, model, nine_type,
                                    current_img)

                # Ready to stop yet?
                if i == cfg['max_iterations']:
                    break

                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass
    return current_img


def make_composite():
    composite = get_start_image().resize((cfg['size'][0] * 2, cfg['size'][1] * 2), Image.LANCZOS)
    composite.save(path)
    order = [0, 2, 6, 8, 1, 3, 5, 7, 4]
    for n in order:
        o_name = get_output_path(n)
        if Path.exists(o_name):
            print_green(f'{o_name} already exists, nice')
            next_img = Image.open(o_name)
            composite = paste_on_base(composite, next_img, n, path)
            composite = paste_on_overlay(composite, next_img, n, cfg['smooth'], path)
        else:
            composite = do_it(n)
    realesrgan = get_realesrgan(cfg['realesrgan_model'], device='cuda')
    composite = super_resolution([composite], realesrgan)[0]
    composite.save(path)
    print_cyan(f'saved {path}')


make_composite()
