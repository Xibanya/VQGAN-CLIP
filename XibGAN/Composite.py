import numpy as np
from scipy.special import comb


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


def center_fade(im, smooth: float = 0.25):
    width, height = im.size
    pixels = im.load()
    for y in range(0, int(height)):
        midpoint = int(height / 2)
        Yalpha = ((midpoint - abs(midpoint - y)) / midpoint)
        Yalpha = smoothstep(smooth, 1, Yalpha)
        for x in range(0, int(width)):
            midpoint = int(width / 2)
            Xalpha = ((midpoint - abs(midpoint - x)) / midpoint)
            Xalpha = smoothstep(smooth, 1, Xalpha)
            alpha = Xalpha * Yalpha
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
    filename = f'{output_dir}/{name}'
    base.save(f'{filename}.png')
    return base


def paste_on_overlay(overlay, image, i, smooth, output_dir, name):
    w, h = image.size
    half_width = int(w / 2)
    image.putalpha(255)
    if i == 4:
        img4Mask = center_fade(image, smooth=smooth)
        overlay.paste(image, (half_width, half_width), mask=img4Mask)
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
            overlay.paste(image, (half_width, w if i > 1 else 0), mask=a_mask)
        else:
            a_mask = vertical_fade(image, smooth=smooth)
            overlay.paste(image, (w if i > 1 else 0, half_width), mask=a_mask)

    overlay.save(f'{output_dir}/{name}.png')
    return overlay
