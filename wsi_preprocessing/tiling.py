# print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# print("                                                        Image Tiling")
# print("")
# print("* Version          : v1.0.1")
# print("")
# print("* Last update      : 2023-09-13")
# print("* Written by       : Francesco Cisternino")
# print("* Edite by         : Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song.")
# print("")
# print("* Description      : Some utilities for the segmentation pipeline.")
# print("")
# print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# import general packages
import os

# for argument parser
import argparse
import textwrap

import datetime
import numpy as np
import openslide
from PIL import Image
from PIL import ImageDraw

# import custom packages/functions
from segmentation_utils import select_random_tiles, get_chunk_wsi
from segmentation_utils import slide_to_scaled_pil_image, save_hdf5, get_chunk
from dataset import Whole_Slide_Bag
from torch.utils.data import DataLoader


def get_args_parser():
    # GENERAL
    parser = argparse.ArgumentParser("Tiling", add_help=False)

    # JOB INDEX
    parser.add_argument(
        "--index",
        default=0,
        help="Index of actual job to split the workload; the default is `0`.",
    )

    # NUMBER OF TASKS
    parser.add_argument(
        "--num_tasks",
        default=1,
        type=str,
        help="The number of tasks, i.e. jobs to split the workload; the default is `1`.",
    )

    # SEGMENTATION OUTPUT DIRECTORY
    parser.add_argument(
        "--output_dir",
        default="./PROCESSED/",
        type=str,
        help="The root output directory containing all the output results (to be created); the default is `./PROCESSED/`.",
    )

    # MASKS DIR
    parser.add_argument(
        "--masks_dir",
        default="./PROCESSED/masks/",
        type=str,
        help="The output subdirectory where the black and white masks will be stored; the default is `./PROCESSED/masks/`.",
    )

    # ADD TISSUE NAME TO MASKS DIR
    parser.add_argument(
        "--masks_dir_no_tissue_name",
        default=False,
        action="store_true",
        help="Whether to add tissue_name to the default masks folder `./PROCESSED/masks/`.",
    )

    # BATCH SIZE (tiles level)
    parser.add_argument(
        "--batch_size_tiling",
        default=512,
        type=int,
        help="The batch size for the tiling process; the default is `512`.",
    )

    # SAVE THUMBNAILS
    parser.add_argument(
        "--save_thumbnails",
        default=True,
        help="Whether to save segmentation thumbnails; the default is `True`.",
    )

    return parser


def tile_slides(args, chunk):
    print("Jobs starting...")
    # === DATA === #
    slides = chunk

    h5_dir = os.path.join(args.output_dir, "patches")
    os.makedirs(h5_dir, exist_ok=True)

    if args.save_thumbnails:
        thumbnails_dir = os.path.join(args.output_dir, "thumbnails")
        os.makedirs(thumbnails_dir, exist_ok=True)

    # tile size
    tile_size = args.tile_size
    # scale factor
    SCALE = 32

    count = 0
    # Iteration over list of slidenames
    for slidename in slides:
        print(slidename, flush=True)

        # Open the slide
        slide = openslide.open_slide(slidename)
        # Slide name without extension
        slidename_noext = slidename.split("/")[-1].rsplit(".", 1)[0]
        # Tissue
        tissue_name = slidename.split("/")[-2]
        # Read the mask
        if args.masks_dir_no_tissue_name:
            if not os.path.exists(
                os.path.join(args.masks_dir, slidename_noext + ".jpg")
            ):
                print(f"Mask not found for {slidename_noext}")
                continue
            mask = Image.open(os.path.join(args.masks_dir, slidename_noext + ".jpg"))
        else:
            if not os.path.exists(
                os.path.join(args.masks_dir, tissue_name, slidename_noext + ".jpg")
            ):
                print(f"Mask not found for {slidename_noext}")
                continue
            mask = Image.open(
                os.path.join(args.masks_dir, tissue_name, slidename_noext + ".jpg")
            )
        mask = np.array(mask)
        # Whole Slide Bag Dataset
        slide_dataset = Whole_Slide_Bag(slide, tile_size=args.tile_size, mask=mask)
        # Tiles Loader
        # You could get the following message:
        # UserWarning: This DataLoader will create 4 worker processes in total. Our suggested
        # max number of worker in current system is 2, which is smaller than what this
        # DataLoader is going to create. Please be aware that excessive worker creation might
        # get DataLoader running slow or even freeze, lower the worker number to avoid
        # potential slowness/freeze if necessary.
        # It's just a warning, but perhaps it is good to take note and keep `num_workers=2`
        # tiles_loader = DataLoader(dataset=slide_dataset, batch_size=300, num_workers=2)
        tiles_loader = DataLoader(dataset=slide_dataset, batch_size=300, num_workers=0)
        # Slidename without file extension
        tissue_name = slidename.split("/")[-2]

        # Downscaled version of the slide
        downscaled_img, _ = slide_to_scaled_pil_image(slide, SCALE_FACTOR=SCALE)
        draw = ImageDraw.Draw(downscaled_img)
        coords = []
        start = datetime.datetime.now()
        print(slidename_noext)
        if os.path.exists(os.path.join(h5_dir, slidename_noext + ".coords.h5")):
            print(f"Skipping {slidename_noext}")
            continue

        mode = "w"
        for (col, row), res in tiles_loader:

            tissue_indexes = (res[:] > 0.3).nonzero(as_tuple=False)

            for t_idx in list(tissue_indexes):

                coords = np.array([int(col[t_idx]), int(row[t_idx])])
                asset_dict = {"coords": np.array([coords])}
                output_path = os.path.join(h5_dir, slidename_noext + ".coords.h5")
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)

                mode = "a"

                if args.save_thumbnails:
                    s = (int(coords[0] / SCALE), int(coords[1] / SCALE))
                    draw.rectangle(
                        (
                            (s[0], s[1]),
                            (s[0] + tile_size / SCALE, s[1] + tile_size / SCALE),
                        ),
                        fill=None,
                        outline="green",
                        width=2,
                    )

        end = datetime.datetime.now()
        print(f"Time required for slide {slidename}: {end - start}", flush=True)
        if args.save_thumbnails:
            downscaled_img.save(
                os.path.join(thumbnails_dir, slidename_noext + "_segm.png")
            )
        count += 1
        print(f"Tiling performed for {count}/{len(chunk)} slides", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tiling", parents=[get_args_parser()])
    args = parser.parse_args()
    # chunk = get_chunk('/group/glastonbury/GTEX-subset/', int(args.index), int(args.num_tasks))
    chunk = get_chunk("", int(args.index), int(args.num_tasks))
    tile_slides(args, chunk)

# print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# print("+ The MIT License (MIT)                                                                                               +")
# print("+ Copyright (c) 2023 Francesco Cisternino | Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song +")
# print("+                                                                                                                     +")
# print("+ Permission is hereby granted, free of charge, to any person obtaining a copy of this software and                   +")
# print("+ associated documentation files (the \"Software\"), to deal in the Software without restriction, including           +")
# print("+ without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell             +")
# print("+ copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the            +")
# print("+ following conditions:                                                                                               +")
# print("+                                                                                                                     +")
# print("+ The above copyright notice and this permission notice shall be included in all copies or substantial                +")
# print("+ portions of the Software.                                                                                           +")
# print("+                                                                                                                     +")
# print("+ THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT             +")
# print("+ LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO           +")
# print("+ EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER           +")
# print("+ IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR             +")
# print("+ THE USE OR OTHER DEALINGS IN THE SOFTWARE.                                                                          +")
# print("+                                                                                                                     +")
# print("+ Reference: http://opensource.org.                                                                                   +")
# print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
