import kornia.augmentation as K
import yaml
import torch
from .Utils import resample, clamp_with_grad
from torch import nn

CONFIG_DIRECTORY = 'Config'
AUGMENT_CONFIG = 'augment_config.yaml'
DEFAULT_AUGMENTS = [['Af', 'Pe', 'Ji', 'Er']]

with open(f"{CONFIG_DIRECTORY}/{AUGMENT_CONFIG}", "r") as f:
    afg = yaml.load(f, Loader=yaml.FullLoader)


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


augments = afg['augments'] if afg['augments'] else []
if not augments and not afg['no_augments']:
    augments = DEFAULT_AUGMENTS



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


def get_affine(cut_method):
    cm = cut_method
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
        get_affine('updatedpooling'),
        K.RandomPerspective(distortion_scale=0.7, p=0.7)
    ]
    return augment_list


def get_augment_list(cut_method, cut_size):
    augment_list = []
    cm = cut_method

    if afg['no_augments']:
        if cm == 'updatedpooling':
            augment_list.append(get_color_jitter())
            augment_list.append(get_erasing())
            augment_list.append(get_affine(cut_method))
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
        augment_list.append(get_affine(cut_method))
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
        augment_list = get_augment_list('latest', cut_size)
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
        augment_list = get_augment_list('nrupdated', cut_size)

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
            K.RandomAffine(degrees=30, translate=(0.1, 0.1), p=0.8, padding_mode='border'),
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


def get_cutout_function(cfg, cut_size):
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
    return make_cutouts
