# print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# print("                                                 Segmentation Utilities")
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

import math
import h5py
import PIL
import os
import glob
import random


def get_chunk(data_folder, i, num_tasks):
    ROOT_DIR = data_folder
    slides = []
    tissues_list = os.listdir("./")  # os.listdir('/group/glastonbury/GTEX-subset/')
    for tissue in tissues_list:
        list = glob.glob(os.path.join(data_folder, tissue, "*.svs"))
        for elem in list:
            slides.append(elem)

    print("Number of slides:", len(slides), flush=True)
    slides_per_job = math.ceil(len(slides) / num_tasks)
    chunks = [
        slides[x : x + slides_per_job] for x in range(0, len(slides), slides_per_job)
    ]
    if i < len(chunks):
        chunk = chunks[i]
        print(f"Chunk {i}: {len(chunk)} slides", flush=True)
    else:
        chunk = []

    return chunk


def slide_to_scaled_pil_image(slide, SCALE_FACTOR=32):
    """
    Convert a WSI slide to a scaled-down PIL image.
    Args:
        slide: An OpenSlide object.
    Returns:
        Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    level = slide.get_best_level_for_downsample(SCALE_FACTOR)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    return img, (large_w, large_h, new_w, new_h)


def save_hdf5(output_path, asset_dict, attr_dict=None, mode="a"):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(
                key,
                shape=data_shape,
                maxshape=maxshape,
                chunks=chunk_shape,
                dtype=data_type,
            )
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0] :] = val
    file.close()
    return output_path


def select_random_tiles(
    idx, num_tasks, TISSUES_DIR="./"
):  # /group/glastonbury/gtex/gtex_images/*/
    import random

    tissues = {}
    slides = []
    for folder in glob.glob(TISSUES_DIR):
        tissue_name = folder.split("/")[-2]
        tissue_slides = glob.glob(os.path.join(folder, "*.svs"))
        random.shuffle(tissue_slides)
        for j in range(5):
            slides.append(tissue_slides[j])

    print("Number of slides:", len(slides), flush=True)
    slides_per_job = math.ceil(len(slides) / num_tasks)
    chunks = [
        slides[x : x + slides_per_job] for x in range(0, len(slides), slides_per_job)
    ]
    if idx < len(chunks):
        chunk = chunks[idx]
        print(f"Chunk {idx}: {len(chunk)} slides", flush=True)
    else:
        chunk = []

    return chunk


def get_chunk_wsi(idx, num_tasks, dir):

    chunk = []
    slides = []

    if os.path.exists(os.path.join(dir, "_ndpi")):
        print("_ndpi found")
        slides += glob.glob(os.path.join(dir, "_ndpi/*.ndpi"))
        slides += glob.glob(os.path.join(dir, "_ndpi/_images/*.ndpi"))
    if os.path.exists(os.path.join(dir, "_tif")):
        print("_tif found")
        slides += glob.glob(os.path.join(dir, "_tif/*.TIF"))
    slides += glob.glob(os.path.join(dir, "_tif/_images/*.TIF"))

    if os.path.exists(os.path.join(dir, "_images_dropzone")):
        print("_images_dropzone found")
        slides += glob.glob(os.path.join(dir, "_images_dropzone/*.ndpi"))
        slides += glob.glob(os.path.join(dir, "_images_dropzone/*.TIF"))
    if os.path.exists(dir):
        print("Slide dir found: ", dir)
        slides += glob.glob(f"{dir}/*.ndpi")
        slides += glob.glob(f"{dir}/*.TIF")

    print("Number of slides:", len(slides), flush=True)
    slides_per_job = math.ceil(len(slides) / num_tasks)
    chunks = [
        slides[x : x + slides_per_job] for x in range(0, len(slides), slides_per_job)
    ]
    chunk = chunks[idx]
    print(f"Chunk {idx}: {len(chunk)} slides", flush=True)

    # debug
    try:
        chunk = chunks[idx]
    except IndexError as e:
        print(f"IndexError: {e}")
        print(f"idx: {idx}, len(chunks): {len(chunks)}")
        # Add more debug information as needed
        raise  # Reraise the exception to see the full traceback

    return chunk


def get_coords_h5(filename):
    with h5py.File(filename, "r") as f:
        # get first object name/key; may or may NOT be a group
        coords_group_key = list(f.keys())[0]
        coords = f[coords_group_key][()]  # returns as a numpy array
        return coords


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
