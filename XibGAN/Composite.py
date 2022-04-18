import numpy as np
from PIL import Image
from scipy.special import comb
from datetime import datetime
from .colors import print_cyan


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


def horizontal_fade(im, smooth: float = 0.25):
    width, height = im.size
    pixels = im.load()
    for x in range(0, int(width)):
        midpoint = int(width / 2)
        alpha = ((midpoint - abs(midpoint - x)) / midpoint)
        alpha = smoothstep(smooth, 1, alpha)
        for y in range(0, int(height)):
            pixels[x, y] = pixels[x, y][:3] + (int(alpha * 255),)
    return im


def vertical_fade(im, smooth: float = 0.25):
    width, height = im.size
    pixels = im.load()
    for y in range(0, int(height)):
        midpoint = int(height / 2)
        alpha = ((midpoint - abs(midpoint - y)) / midpoint)
        alpha = smoothstep(smooth, 1, alpha)
        for x in range(0, int(width)):
            pixels[x, y] = pixels[x, y][:3] + (int(alpha * 255),)
    return im


def paste_on_base(base, image, i, output_dir, name):
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
    w, h = image.size
    x = w if (i + 1) % 2 == 0 else 0
    y = w if i > 1 else 0
    base.paste(image, (x, y))
    filename = f'{output_dir}/{name}_Base'
    base.save(f'{filename}.png')
    print_cyan(f'img {i} pasted on {filename}')
    return base


# assuming all nine images passed in
def get_base(images):
    images = [images[0], images[2], images[6], images[8]]
    w, h = images[0].size
    size = w * 2
    base = Image.new("RGB", (size, size))
    for i in range(len(images)):
        base.paste(images[i],
                   (w if (i + 1) % 2 == 0 else 0,
                    w if i > 1 else 0))
    return base


# assuming all nine images passed in
def get_overlay(
        images: list[Image],
        smooth: float,
        save: bool = False,
        name: str = None,
        output_dir: str = None
):
    img4 = images[4]
    images = [images[1], images[3], images[5], images[7]]
    w, h = images[0].size
    size = w * 2
    half_width = int(w / 2)
    overlay = Image.new("RGBA", (size, size))
    overlay.putalpha(0)
    for i in range(len(images)):
        if i == 0 or i == 3:
            a_mask = horizontal_fade(images[i], smooth=smooth)
            overlay.paste(images[i], (half_width, w if i > 1 else 0), mask=a_mask)
        else:
            a_mask = vertical_fade(images[i], smooth=smooth)
            overlay.paste(images[i], (w if i > 1 else 0, half_width), mask=a_mask)

    if img4 is not None:
        img4Mask = horizontal_fade(img4, smooth)
        img4Mask = vertical_fade(img4Mask, smooth)
        overlay.paste(img4, (half_width, half_width), mask=img4Mask)
    if save:
        now = datetime.now().strftime("%dT%H-%M-%S")
        overlay.save(f'{output_dir}/{name}_Overlay_{now}.png')
    return overlay
