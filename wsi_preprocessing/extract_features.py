#!/usr/bin/env python3
# -*- coding: utf-8 -*-
print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)
print("                                                  Feature extraction")
print("")
print("* Version          : v1.0.0")
print("")
print("* Last update      : 2023-10-20")
print("* Written by       : Francesco Cisternino")
print(
    "* Edite by         : Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song."
)
print("")
print(
    "* Description      : Feature extraction pipeline to identify features in segmented whole-slide images (WSI) "
)
print("                     using the U-Net model.")
print("")
print("                     [1] https://github.com/MaryamHaghighat/PathProfiler")
print("")
print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)

import os
import torch
import argparse
import textwrap
import datetime
import numpy as np
import glob
import openslide
import math
from segmentation_utils import get_coords_h5
from segmentation_utils import save_hdf5
from model import FeaturesExtraction, FeaturesExtraction_IMAGENET
from dataset import Tiles_Bag
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import shutil
import os
import h5py


# Arguments
def get_args_parser():
    parser = argparse.ArgumentParser("Feature extraction", add_help=False)

    # JOB INDEX
    parser.add_argument("-index", type=str, default=0, help="Index of actual job.")

    # NUMBER OF TASKS
    parser.add_argument("-num_tasks", type=str, default=1, help="Number of tasks.")

    # DATA DIRECTORY
    parser.add_argument(
        "-h5_data",
        type=str,
        default="/hpc/dhl_ec/VirtualSlides/STAIN/_images/patches_512",
        help="Path to directory containing h5 coordinates files.",
    )

    # SLIDES DIRECTORY
    parser.add_argument(
        "-slide_folder",
        type=str,
        default="/hpc/dhl_ec/VirtualSlides/STAIN/_images",
        help="Path to directory containing the slides. This is the directory that contains the _images folder. If -slides is provided, this argument is ignored.",
    )

    # SPECIFY SLIDES
    parser.add_argument(
        "-slides",
        type=str,
        nargs="+",
        help="Provide specific slides to process. If this argument is provided, -slide_folder is ignored.",
    )

    # FEATURES FILES SAVE DIR
    parser.add_argument(
        "-output_dir",
        type=str,
        default="/hpc/dhl_ec/VirtualSlides/STAIN/_images/features",
        help="Path to features .h5/.pt storage.",
    )

    # FEATURES EXTRACTION CHECKPOINT
    parser.add_argument(
        "-features_extraction_checkpoint",
        type=str,
        default="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/checkpoints/checkpoint_ViT_AT.pth",
        help="Checkpoint for the tiles features extraction model.",
    )

    # BATCH SIZE (tiles level)
    parser.add_argument(
        "-batch_size",
        type=int,
        default=1,
        help="Batch size (tiles); the default is `1`",
    )

    # TILE SIZE
    parser.add_argument(
        "-tile_size",
        type=int,
        default=512,
        help="Tile size (related to the number of tiles that are processed by the network); the default is `512`.",
    )

    # SAVE FEATURES
    parser.add_argument(
        "-save_features", default=False, help="Whether to save segmentation thumbnails."
    )

    # SAVE TILES
    parser.add_argument(
        "-save_tiles", default=True, help="Whether to save tiles images."
    )

    # DEBUG
    parser.add_argument(
        "-verbose",
        default=False,
        help="Whether to print verbose messages for debugging.",
    )

    # VERSION
    parser.add_argument(
        "-version",
        action="version",
        version="%(prog)s v1.0.0-2023-10-24",
        help="Show program's version number and exit.",
    )

    return parser


# EXTRACT FEATURES
def extract_features(args, chunk):
    print("Jobs starting...")
    # === DATA === #
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # === FEATURES EXTRACTION MODEL === #
    features_extraction_class = FeaturesExtraction_IMAGENET(args)
    features_extraction = features_extraction_class.model
    features_extraction.eval()
    features_extraction.to(DEVICE)

    # === OUTPUT DIR === #
    # create this when it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    temp_dir = os.path.join(args.output_dir, "temp")

    features_output_dir = os.path.join(args.output_dir)
    features_dir_h5 = os.path.join(features_output_dir, "h5_files")
    features_dir_pt = os.path.join(features_output_dir, "pt_files")

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(features_output_dir, exist_ok=True)
    os.makedirs(features_dir_h5, exist_ok=True)
    os.makedirs(features_dir_pt, exist_ok=True)

    # tile size
    tile_size = args.tile_size

    transform = features_extraction_class.transform
    count = 0
    # Iteration over list of slidenames
    for slidename in chunk:

        print(f"Slide {chunk.index(slidename) + 1}/{len(chunk)}", flush=True)

        start = datetime.datetime.now()

        slidename_noext = slidename.split("/")[-1].rsplit(".", 1)[0]

        if os.path.exists(
            os.path.join(features_dir_h5, slidename_noext + ".h5")
        ) and os.path.exists(os.path.join(features_dir_pt, slidename_noext + ".pt")):
            print(f"Features for {slidename_noext} already extracted", flush=True)
            continue

        # Open the slide
        slide = openslide.open_slide(slidename)
        # Slidename without extension
        # h5 file

        h5_file = os.path.join(args.h5_data, slidename_noext + ".coords.h5")
        # print(h5_file)
        coords = get_coords_h5(h5_file)

        dataset = Tiles_Bag(
            slide=slide, tile_size=args.tile_size, transform=transform, h5=h5_file
        )
        loader = DataLoader(dataset=dataset, batch_size=200, num_workers=2)

        mode = "w"
        f = torch.empty(0, 1000)
        c = torch.empty(0, 2)
        op = os.path.join(features_dir_h5, slidename_noext + ".h5")
        for tile, coords in loader:
            col, row = coords
            # extract features
            features = features_extraction_class.extractFeatures(tile, device=DEVICE)
            c = torch.cat(
                (c, torch.concat((col.unsqueeze(0), row.unsqueeze(0)), dim=0).T)
            )
            f = torch.cat((f, torch.from_numpy(features)))

        features = f
        coords = c
        with h5py.File(op, "w") as fi:
            # Store arrays in HDF5 file with specified names
            fi.create_dataset("coords", data=coords.numpy())
            fi.create_dataset("features", data=features.numpy())

        torch.save(features, os.path.join(features_dir_pt, slidename_noext + f".pt"))
        end = datetime.datetime.now()

        print(f"Time required: {end - start}, shape: {features.shape}", flush=True)


# MAIN
if __name__ == "__main__":
    # parser = argparse.ArgumentParser('features-extraction', parents=[get_args_parser()])
    parser = argparse.ArgumentParser(
        "Feature extraction",
        parents=[get_args_parser()],
        description="This script will extract features from segmented whole-slide images (WSI), for example .TIF- or .ndpi-files, from (a list of given) images.",
        usage="extraction_feature.py -index -num_tasks -h5_data [-slide_folder | -slides] -output_dir -features_extraction_checkpoint -batch_size -tile_size -save_features -save_tiles; optional: for help: -h/--help; for verbose (with extra debug information): -verbose; for version information: -version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            "Copyright (c) 2023 Francesco Cisternino | Craig Glastonbury | Sander W. van der Laan (s.w.vanderlaan-2@umcutrecht.nl) | Clint L. Miller | Yipei Song"
        ),
    )
    args = parser.parse_args()

    if args.slides is not None:
        # If specific slides are provided, use them instead of the slide_folder
        files = args.slides
        if args.verbose:
            print("VERBOSE <<>> Files found:", files)
    else:
        if args.verbose:
            print(
                "VERBOSE <<>> Checking existence of slides in directory [",
                args.slide_folder,
                "]",
                flush=True,
            )
        files = []
        if os.path.exists(os.path.join(args.slide_folder, "_ndpi")):
            if args.verbose:
                print("VERBOSE <<>> _ndpi found")
            files += glob.glob(os.path.join(args.slide_folder, "_ndpi/*.ndpi"))
            files += glob.glob(os.path.join(args.slide_folder, "_ndpi/_images/*.ndpi"))
        if os.path.exists(os.path.join(args.slide_folder, "_tif")):
            if args.verbose:
                print("VERBOSE <<>> _tif found")
            files += glob.glob(os.path.join(args.slide_folder, "_tif/*.TIF"))
            files += glob.glob(os.path.join(args.slide_folder, "_tif/_images/*.TIF"))
        # image_folder = os.path.join(args.slide_folder, '_images')
        # if args.verbose:
        #     print("VERBOSE <<>> Image folder:", image_folder)
        # files = glob.glob(os.path.join(args.slide_folder, '_images/*.TIF')) + glob.glob(os.path.join(args.slide_folder, '_images/*.ndpi'))
        if args.verbose:
            print("VERBOSE <<>> Files found:", files)

    num_tasks = int(args.num_tasks)
    i = int(args.index)
    print("Number of slides found:", len(files), flush=True)
    files_per_job = math.ceil(len(files) / num_tasks)
    chunks = [files[x : x + files_per_job] for x in range(0, len(files), files_per_job)]
    if i < len(chunks):
        chunk = chunks[i]
        print(f"Chunk {i}: {len(chunk)} slides", flush=True)
    else:
        chunk = []
    extract_features(args, chunk)

print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)
print(
    "+ The MIT License (MIT)                                                                                               +"
)
print(
    "+ Copyright (c) 2023 Francesco Cisternino | Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song +"
)
print(
    "+                                                                                                                     +"
)
print(
    "+ Permission is hereby granted, free of charge, to any person obtaining a copy of this software and                   +"
)
print(
    '+ associated documentation files (the "Software"), to deal in the Software without restriction, including           +'
)
print(
    "+ without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell             +"
)
print(
    "+ copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the            +"
)
print(
    "+ following conditions:                                                                                               +"
)
print(
    "+                                                                                                                     +"
)
print(
    "+ The above copyright notice and this permission notice shall be included in all copies or substantial                +"
)
print(
    "+ portions of the Software.                                                                                           +"
)
print(
    "+                                                                                                                     +"
)
print(
    '+ THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT             +'
)
print(
    "+ LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO           +"
)
print(
    "+ EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER           +"
)
print(
    "+ IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR             +"
)
print(
    "+ THE USE OR OTHER DEALINGS IN THE SOFTWARE.                                                                          +"
)
print(
    "+                                                                                                                     +"
)
print(
    "+ Reference: http://opensource.org.                                                                                   +"
)
print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)
