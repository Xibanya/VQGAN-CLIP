# XibQGAN-CLIP Overview

Make high res VQGAN output by compositing a bunch of lower res outputs!

Anyone who has messed with any pytorch/tensorflow stuff locally has quickly run into the dreaded OOM (out of memory). Even with a really beefy machine, the size and quality of generated images is pretty low. The upper bound for a top-of-the-line consumer-grade PC is around 512px. One way to generate larger images on limited hardware is to use the CPU rather than the GPU, but that takes forever, like literal days. I'm way too impatient for that, so I came up with this variant: generate a buncha smaller images then blend them together into a bigger one! This allows generating output much bigger than would ordinarily be possible on consumer-grade hardware, as well as generating output of the same size as usual but way faster.

The quality of the output generated is probably not going to be quite as good as if you'd generated an entire image at the desired resolution, but doing it like this is really fast!! Which is great if you're less interested in generating super impressive images to wow others and more interested in entertaining yourself by having the AI make you cool pictures.

## Installing
see [the original repo](https://github.com/nerdyrodent/VQGAN-CLIP) for more detailed installation steps.

first, you go to wherever you want all this crap, then in the command line put
```shell
git clone https://github.com/Xibanya/VQGAN-CLIP.git
```
then enter the directory that gets made for the rest of the setup.

you need to install pytorch. if you already have it but you didn't install it with CUDA, you have to uninstall it and reinstall it again like this:
```sh
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
Check the official site for the install instruction for your specific OS and environment: https://pytorch.org/get-started/locally/

you can also install the other stuff you probably need with
```sh
pip install -r requirements.txt
```

then you gotta get the actual VQGAN stuff

```shell
git clone 'https://github.com/nerdyrodent/VQGAN-CLIP'
cd VQGAN-CLIP
git clone 'https://github.com/openai/CLIP'
git clone 'https://github.com/CompVis/taming-transformers'
```

then get a pretrained model. This is a big file so the download might take a few minutes.

```sh
mkdir checkpoints

curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
```

There's also a `download_models.sh` script you can tweak for downloading other models, including one of mine! If you don't edit it, it'll download the default imagenet one mentioned above.

## Run
I'm assuming you already know the vagarities of VQGAN+CLIP, so I'm just gonna cover how to run my fork. 

### Arguments
This version doesn't use command line arguments because it's intended to be run from your IDE. To customize behavior, change the settings in the files in the Config folder. These have been commented pretty thoroughly in the files themselves

Past the use of the config file rather than the command line arguments, `generate.py` works pretty much the same as in the main branch (although I expose a lot more options in the config file than the CLI arguments available there.) Everything else I'm going to write about is going to pertain to my composite image generation technique, whose main script is `generate_loop.py`

### Slices
The composite is made by blending nine slices of the original image.

``` 
 ____________________
|      |      |      |
| NW:0 |  N:1 | NE:2 |
|______|______|______|
|      |      |      |
|  W:3 |  X:4 |  E:5 |
|______|______|______|
|      |      |      |
| SW:6 |  S:7 | SE:8 |
|______|______|______|
```

`generate_loop.py` will generate every slice of the composite. If `save_nine` is True, each individual slice will also be saved to disk; otherwise only the composite will be saved to disk. Individual slices are saved with the naming schema index + name of initial image, so the northeast slice of vapor.png, for example, would be saved as `2_vapor.png`

`generate.py` will generate a single slice if an integer from 0 to 8 or a two letter cardinal/ordinal direction is assigned to the `nine_type` config option. (`nine_type` isn't used in `generate_loop.py` because it generates every slice.)

if an image with the expected naming schema for a slice already exists when `generate_loop.py` is run, that image will be blended into the composite without being regenerated. So if your run gets interrupted, and you were saving the slices along the way (`save_nine: True` in `Config/config.yaml`), you can pick back up where you left off when you run again; this also lets you regenerate specific slices with `generate.py` and remake the composite without generating every single slice again.

### Prompts
The whole point of this is to iterate on an input image at a higher res than is normally possible, so at a minimum you have to have an initial image. I have provided a test image, `vapor.png`. When putting together the path to the initial image, the main function will concatenate `input_dir` with `init_image`, so init_image can just be the file name of the image in the input folder without any directory path stuff in front of it.

the `prompts` entry in config.yaml works just like the prompts command line argument; you can separate different prompts with `|` and specify different weights with `:5`, `:-0.3`, etc. example: `city skyline|synthwave:0.75|photo:-1`

the `image_prompts` entry also works just like the original; multiple images must be separated with pipes, weights specified with colons

`slice_prompts` is unique to this variation; these are additional prompts that are appended for just the slice currently being processed. If not every slice of the image has the same thing in it, you can often get a much better result by describing what's just in that chunk there. The `config.yaml` file in this repo has these filled out as an example. You can also leave them blank, just make sure that there are nine entries in that array total (so if you have nothing extra to add for index 4, for example, that would just be a `-` on that line in the yaml file)

### Size
the `size` argument is the resolution of the images generated with pytorch, so when using `generate_loop.py`, that is the resolution of each slice, the final composite will be twice that size. So, without upscaling, if the size is set to 256, you will get a 512px composite. The output of `generate.py` will be the specified size directly (excluding upscaling)

I have added the option to upscale the final output with realesrgan, this is implemented in both `generate_loop.py` and `generate.py`


### Augments
Modify the `Config/augment_config.yaml` file to specify augments (if any). Way more options are surfaced here than in the original CLI implementation. These augments are mostly calls to the [Kornia API](https://github.com/kornia/kornia), so you can check out their documentation if you want to really get into how they all work.

### Video
You can't generate a video with `generate_loop.py`, sorry. But you can with `generate.py`; Mess with `Config/video_config.yaml` for the related options 