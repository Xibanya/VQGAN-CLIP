import glob
import os

import numpy as np
from scipy.special import comb
from PIL import Image
from .Utils import resize_image
from .Colors import print_warn


def smoothstep(x_min: float, x_max: float, x: float, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
    result *= x ** (N + 1)
    return result


def lerp(v0: float, v1: float, t: float) -> float:
    return (1 - t) * v0 + t * v1


def inv_lerp(a: float, b: float, v: float) -> float:
    return (v - a) / (b - a)


def horizontal_lerp(im, smooth):
    width, height = im.size
    pixels = im.load()
    for x in range(0, int(width)):
        midpoint = int(width / 2)
        alpha = ((midpoint - abs(midpoint - x)) / midpoint)
        alpha = smoothstep(smooth[0], smooth[1], alpha)
        for y in range(0, int(height)):
            pixels[x, y] = pixels[x, y][:3] + (int(alpha * 255),)
    return im


def horizontal_fade(im, smooth):
    width, height = im.size
    pixels = im.load()
    for x in range(0, int(width)):
        midpoint = int(width / 2)
        alpha = ((midpoint - abs(midpoint - x)) / midpoint)
        alpha = smoothstep(smooth[0], smooth[1], alpha)
        for y in range(0, int(height)):
            pixels[x, y] = pixels[x, y][:3] + (int(alpha * 255),)
    return im


def vertical_fade(im, smooth):
    width, height = im.size
    pixels = im.load()
    for y in range(0, int(height)):
        midpoint = int(height / 2)
        alpha = ((midpoint - abs(midpoint - y)) / midpoint)
        alpha = smoothstep(smooth[0], smooth[1], alpha)
        for x in range(0, int(width)):
            pixels[x, y] = pixels[x, y][:3] + (int(alpha * 255),)
    return im


def center_fade(im, smooth):
    width, height = im.size
    pixels = im.load()
    for y in range(0, int(height)):
        midpoint = int(height / 2)
        Yalpha = ((midpoint - abs(midpoint - y)) / midpoint)
        Yalpha = smoothstep(smooth[0] - 0.1, min(1, smooth[1] + 0.1), Yalpha)
        for x in range(0, int(width)):
            midpoint = int(width / 2)
            Xalpha = ((midpoint - abs(midpoint - x)) / midpoint)
            Xalpha = smoothstep(smooth[0] - 0.1,  min(1, smooth[1] + 0.1), Xalpha)
            alpha = Xalpha * Yalpha
            pixels[x, y] = pixels[x, y][:3] + (int(alpha * 255),)
    return im


def paste_on_base(base, image, i, path):
    if i == 0:
        i = 0
    elif i == 2:
        i = 1
    elif i == 6:
        i = 2
    elif i == 8:
        i = 3
    else:
        return base

    baseWidth, baseHeight = base.size
    w, h = image.size
    if w * 2 != baseWidth:
        w = int(baseWidth / 2)
        image = image.resize((w, w), Image.BILINEAR)

    x = w if (i + 1) % 2 == 0 else 0
    y = w if i > 1 else 0
    base.paste(image, (x, y))
    base.save(path)
    return base


def paste_on_overlay(overlay, image, i, smooth, path):
    w, h = image.size
    baseWidth, baseHeight = overlay.size
    if w * 2 != baseWidth:
        w = int(baseWidth / 2)
        image = image.resize((w, w), Image.BILINEAR)

    half_width = int(w / 2)
    image.putalpha(255)
    if i == 4:
        img4Mask = center_fade(image, smooth=smooth)
        overlay.alpha_composite(img4Mask, (half_width, half_width))
    elif i == 0 or i % 2 == 0:
        return overlay
    else:
        if i == 1:
            i = 0
        elif i == 3:
            i = 1
        elif i == 5:
            i = 2
        elif i == 7:
            i = 3
        if i == 0 or i == 3:
            a_mask = horizontal_fade(image, smooth=smooth)
            overlay.alpha_composite(a_mask, (half_width, w if i > 1 else 0))
        else:
            a_mask = vertical_fade(image, smooth=smooth)
            overlay.alpha_composite(a_mask, (w if i > 1 else 0, half_width))

    overlay.save(path)
    return overlay


def get_base(images):
    if len(images) == 9:
        images = [images[0], images[2], images[6], images[8]]
    w, h = images[0].size
    size = w * 2
    base = Image.new("RGBA", (size, size))
    for i in range(len(images)):
        base.paste(images[i],
                   (w if (i + 1) % 2 == 0 else 0,
                    w if i > 1 else 0))
    return base


def make_base(name: str, folder: str):
    paths = []
    types = (f'00_{name}*.png', f'02_{name}*.png', f'06_{name}*.png', f'08_{name}*.png')
    for ext in types:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    images = []
    for p in paths:
        images.append(Image.open(p))

    base = get_base(images)
    return base


def get_overlay(images, smooth):
    img4 = images[4]
    images = [images[1], images[3], images[5], images[7]]
    w, h = images[0].size
    size = w * 2
    half_width = int(w / 2)
    overlay = Image.new("RGBA", (size, size))
    overlay.putalpha(0)
    overlay_mask = Image.new("RGBA", (size, size))
    overlay_mask.putalpha(0)
    for i in range(len(images)):
        if i == 0 or i == 3:
            overlay.paste(images[i], (half_width, w if i > 1 else 0))
            a_mask = horizontal_fade(images[i], smooth=smooth)
            overlay_mask.paste(images[i], (half_width, w if i > 1 else 0), mask=a_mask)
        else:
            overlay.paste(images[i], (w if i > 1 else 0, half_width))
            a_mask = vertical_fade(images[i], smooth=smooth)
            overlay_mask.paste(images[i], (w if i > 1 else 0, half_width), mask=a_mask)

    if img4 is not None:
        img4Mask = horizontal_fade(img4, smooth)
        img4Mask = vertical_fade(img4Mask, smooth)
        overlay.paste(img4, (half_width, half_width))
        overlay_mask.paste(img4, (half_width, half_width), mask=img4Mask)
    return overlay, overlay_mask


def make_overlay(name: str, folder: str, smooth):
    paths = []
    types = (f'0*_{name}*.png', f'__{name}*.png')
    for ext in types:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    images = []
    for p in range(len(paths)):
        im = Image.open(paths[p])
        im.putalpha(255)
        images.append(im)
    overlay = get_overlay(images, smooth)
    return overlay


def combined(name: str, folder: str, smooth, path: str):
    paths = []
    images = []
    types = (f'0*_{name}*.png', f'__{name}*.png')
    for ext in types:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    order = [0, 2, 6, 8, 1, 3, 5, 7, 4]
    for p in paths:
        images.append(Image.open(p))
    w, h = images[0].size
    size = int(w * 2)
    base = Image.new("RGBA", (size, size))
    for n in order:
        next_img = Image.open(paths[n])
        base = paste_on_base(base, next_img, n, path)
        base = paste_on_overlay(base, next_img, n, smooth, path)
    composite = Image.new("RGB", base.size, (255, 255, 255))
    composite.paste(base, mask=base.split()[3])
    return composite
