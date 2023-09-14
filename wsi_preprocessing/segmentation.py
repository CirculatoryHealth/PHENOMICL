# print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# print("                                                     Segmentation")
# print("")
# print("* Version          : v1.0.1")
# print("")
# print("* Last update      : 2023-09-13")
# print("* Written by       : Francesco Cisternino")
# print("* Edite by         : Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song.")
# print("")
# print("* Description      : Segmentation pipeline to identify tissue and mask non-tissue areas using the U-Net model.")
# print("                     The output is a black and white mask.")
# print("")
# print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# import general packages
import sys
# set the path 
sys.path.extend(["../.", "."])

from matplotlib.cbook import ls_mapper
import time
import os
import gc
import numpy as np
import glob
import decimal

# for argument parser
import argparse
import textwrap

# import openslide and cv2
import cv2
import openslide

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import scipy.signal
from skimage.morphology import remove_small_objects

# import custom packages/functions
from tiling import tile_slides
import tiling
from segmentation_utils import get_chunk_AE

# import model
from PathProfiler.common.wsi_reader import get_reader_impl
from PathProfiler.tissue_segmentation.unet import UNet

parser = argparse.ArgumentParser(parents=[tiling.get_args_parser()],
                                 prog='segmentation',
	description='This script will segment whole-slide images (.TIF or .ndpi) into tissue and non-tissue, and create masked images at a given level of magnification from (a list of given) images.',
	usage='segmentation.py --slide_id --model --mask_magnification --mpp_level_0 --gpu_id --tile_size --batch_size; optional: --slide_dir ; for help: -h/--help',
	formatter_class=argparse.RawDescriptionHelpFormatter,
	epilog=textwrap.dedent("Copyright (c) 2023 Francesco Cisternino | Craig Glastonbury | Sander W. van der Laan (s.w.vanderlaan-2@umcutrecht.nl) | Clint L. Miller | Yipei Song"))

parser.add_argument('--slide_dir', default='', help='path to WSIs dir', type=str, required=False)
parser.add_argument('--slide_id', default='*', type=str, help='slide filename ("*" for all slides)')
parser.add_argument('--model', default='./PathProfiler/tissue_segmentation/checkpoint_ts.pth', type=str)
parser.add_argument('--mask_magnification', default=2.5, type=float)
parser.add_argument('--mpp_level_0', default=None, type=float)
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--tile_size', default=512, type=int)
parser.add_argument('--batch_size', default=1, type=int)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if not os.path.exists(args.masks_dir):
    os.makedirs(args.masks_dir)


class CLAHE(object):
    # histogram equalisation
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, img):
        HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        HSV[:, :, 0] = self.clahe.apply(HSV[:, :, 0])
        img = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)
        return img


class SegDataset(Dataset):

    def __init__(self, img, patch_size, subdivisions):
        self.img = img
        self.patch_size = patch_size
        self.subdivisions = subdivisions
        self.totensor = transforms.ToTensor()
        self.normalise = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.histeq = CLAHE()
        self.coordinates = self._extract_patches()

    def _extract_patches(self):
        """
        :param img:
        :return: a generator
        """
        step = int(self.patch_size / self.subdivisions)

        row_range = range(0, self.img.shape[0] - self.patch_size + 1, step)
        col_range = range(0, self.img.shape[1] - self.patch_size + 1, step)

        coordinates = []
        for row in row_range:
            for col in col_range:
                coordinates.append((row, col))

        return coordinates

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        row, col = self.coordinates[idx]
        image = self.img[row:(row+self.patch_size), col:(col+self.patch_size), :]

        # instance norm
        image = self.histeq(image)

        # scale between 0 and 1 and swap the dimension
        image = self.totensor(image)
        image = self.normalise(image)

        return image


class TilePrediction(object):
    def __init__(self, patch_size, subdivisions, pred_model, batch_size, workers):
        """
        :param patch_size:
        :param subdivisions: the size of stride is define by this
        :param scaling_factor: what factor should prediction model operate on
        :param pred_model: the prediction function
        """
        self.patch_size = patch_size
        self.subdivisions = subdivisions
        self.pred_model = pred_model
        self.batch_size = batch_size
        self.workers = workers

        self.stride = int(self.patch_size / self.subdivisions)

        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

        self.WINDOW_SPLINE_2D = self._window_2D(window_size=self.patch_size, effective_window_size=patch_size, power=2)

    def _read_data(self, filename):
        """
        :param filename:
        :return:
        """
        mpp2mag = {.25: 40, .5: 20, 1: 10}
        reader = get_reader_impl(filename)
        slide = reader(filename)
        if args.mpp_level_0:
            print('slides mpp manually set to', args.mpp_level_0)
            mpp=slide.properties[openslide.PROPERTY_NAME_MPP_X]
        else:
            try:
                s = openslide.OpenSlide(filename)
                mpp = decimal.Decimal(s.properties[openslide.PROPERTY_NAME_MPP_X])
            except:
                print('slide mpp is not available as "slide.mpp"\n use --mpp_level_0 to enter mpp at level 0 manually.')
        wsi_highest_magnification = mpp2mag[.25 * round(float(mpp) / .25)]
        downsample = wsi_highest_magnification / args.mask_magnification
        slide_level_dimensions = (int(np.round(slide.level_dimensions[0][0]/downsample)),
                                  int(np.round(slide.level_dimensions[0][1]/downsample)))
        img, _ = slide.get_downsampled_slide(slide_level_dimensions, normalize=False)
        img = self._pad_img(img)

        return img

    def _pad_img(self, img):
        """
        Add borders to img for a "valid" border pattern according to "window_size" and
        "subdivisions".
        Image is an np array of shape (x, y, nb_channels).
        """
        aug = int(round(self.patch_size * (1 - 1.0 / self.subdivisions)))
        more_borders = ((aug, aug), (aug, aug), (0, 0))
        ret = np.pad(img, pad_width=more_borders, mode='reflect')

        return ret

    def _unpad_img(self, padded_img):
        """
        Undo what's done in the `_pad_img` function.
        Image is an np array of shape (x, y, nb_channels).
        """
        aug = int(round(self.patch_size * (1 - 1.0 / self.subdivisions)))
        ret = padded_img[aug:-aug, aug:-aug, :]
        return ret

    def _spline_window(self, patch_size, effective_window_size, power=2):
        """
        Squared spline (power=2) window function:
        https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
        """
        window_size = effective_window_size
        intersection = int(window_size / 4)
        wind_outer = (abs(2 * (scipy.signal.triang(window_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)

        aug = int(round((patch_size - window_size) / 2.0))
        wind = np.pad(wind, (aug, aug), mode='constant')
        wind = wind[:patch_size]

        return wind

    def _window_2D(self, window_size, effective_window_size, power=2):
        """
        Make a 1D window function, then infer and return a 2D window function.
        Done with an augmentation, and self multiplication with its transpose.
        Could be generalized to more dimensions.
        """
        # Memoization
        wind = self._spline_window(window_size, effective_window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 2)
        wind = wind * wind.transpose(1, 0, 2)
        return wind

    def _merge_patches(self, patches, padded_img_size):
        """
        :param patches:
        :param padded_img_size:
        :return:
        """
        n_dims = patches[0].shape[-1]
        img = np.zeros([padded_img_size[0], padded_img_size[1], n_dims], dtype=np.float32)

        window_size = self.patch_size
        step = int(window_size / self.subdivisions)

        row_range = range(0, img.shape[0] - self.patch_size + 1, step)
        col_range = range(0, img.shape[1] - self.patch_size + 1, step)

        for index1, row in enumerate(row_range):
            for index2, col in enumerate(col_range):
                tmp = patches[(index1 * len(col_range)) + index2]
                tmp *= self.WINDOW_SPLINE_2D

                img[row:row + self.patch_size, col:col + self.patch_size, :] = \
                    img[row:row + self.patch_size, col:col + self.patch_size, :] + tmp

        img = img / (self.subdivisions ** 2)
        return self._unpad_img(img)

    def batches(self, generator, size):
        """
        :param generator: a generator
        :param size: size of a chunk
        :return:
        """
        source = generator
        while True:
            chunk = [val for _, val in zip(range(size), source)]
            if not chunk:
                raise StopIteration
            yield chunk

    def run(self, filename):
        """
        :param filename:
        :return:
        """

        # read image, scaling, and padding
        padded_img = self._read_data(filename)
        # extract patches
        test_dataset = SegDataset(padded_img, self.patch_size, self.subdivisions)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.workers,
                                                   shuffle=False)
        gc.collect()

        # run the model in batches
        all_prediction = []
        for patches in test_loader:
            if torch.cuda.is_available():
                patches = patches.cuda()

            all_prediction += [self.pred_model(patches).cpu().data.numpy()]

        all_prediction = np.concatenate(all_prediction, axis=0)
        all_prediction = all_prediction.transpose(0, 2, 3, 1)

        result = self._merge_patches(all_prediction, padded_img.shape)

        # confidence
        result = np.argmax(result, axis=2) * 255.0
        result = result.astype(np.uint8)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


def segmentation(chunk):

    #############################################################
    # sanity check
    assert args.mask_magnification in [2.5, 1.25], "tile_magnification should be either 2.5 or 1.25"
    assert os.path.isfile(args.model), "=> no checkpoint found at '{}'".format(args.model)
    #############################################################

    # create model
    unet = UNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = nn.DataParallel(unet).cuda() if torch.cuda.is_available() else nn.DataParallel(unet)
    print("=> loading checkpoint '{}'".format(args.model))
    checkpoint = torch.load(args.model, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.model, checkpoint['epoch']))

    net.eval()
    # You could get the following message:
    # UserWarning: This DataLoader will create 4 worker processes in total. Our suggested 
    # max number of worker in current system is 2, which is smaller than what this 
    # DataLoader is going to create. Please be aware that excessive worker creation might 
    # get DataLoader running slow or even freeze, lower the worker number to avoid 
    # potential slowness/freeze if necessary.
    predictor = TilePrediction(patch_size=args.tile_size,
                               subdivisions=2.0,
                               pred_model=net,
                               batch_size=args.batch_size,
                               workers=2)

    #############################################################################################  

    for slide in chunk:
        print(f'Processing {slide} - {chunk.index(slide)}/{len(chunk)}', flush=True)
        basename = os.path.splitext(os.path.basename(slide))[0]
        tissue_name = slide.split('/')[-2]
        os.makedirs(os.path.join(args.masks_dir, tissue_name), exist_ok=True)
        savename = os.path.join(args.masks_dir, tissue_name, basename + '.jpg')

        if not os.path.exists(savename):
            
            try:
                segmentation = predictor.run(slide)
                segmentation = remove_small_objects(segmentation == 255, 50**2)
                segmentation = (segmentation*255).astype(np.uint8)
                segmentation = cv2.morphologyEx(segmentation, cv2.MORPH_CLOSE, kernel = np.ones((50, 50),np.uint8))
                cv2.imwrite(savename, segmentation)
            except Exception as e:
                print(e,'\nSkipped slide', basename)
                continue

#############################################################################################


if __name__ == '__main__':
    t = time.time()
    DATA_FOLDER = '/hpc/dhl_ec/VirtualSlides/HE' # args.slide_dir # '/hpc/dhl_ec/VirtualSlides/HE'
    chunk = get_chunk_AE( idx= int(args.index), num_tasks=int(args.num_tasks), dir = DATA_FOLDER )
    segmentation(chunk)
    tile_slides(args, chunk)
    print('Tissue segmentation done (%.2f)' % (time.time() - t))

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