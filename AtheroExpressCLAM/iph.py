import argparse
from models.model_clam_sum import CLAM_MB, CLAM_SB
import torch
from utils.eval_utils import initiate_model
from vis_utils.heatmap_utils import slide_to_scaled_pil_image
import h5py
import openslide
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import pandas as pd
from PIL import ImageDraw, ImageFont
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

SCALE_FACTOR = 16

# Generic training settings
parser = argparse.ArgumentParser(description="Configurations for Heatmap Creation")
# data_root_dir => see files organization in README.md
parser.add_argument(
    "--model_type",
    type=str,
    choices=["clam_sb", "clam_mb", "mil"],
    default="clam_sb",
    help="type of model (default: clam_sb, clam w/ single attention branch)",
)
parser.add_argument(
    "--model_size",
    type=str,
    choices=["small", "big", "dino_version", "imagenet"],
    default="dino_version",
    help="size of model, does not affect mil",
)
parser.add_argument(
    "--task",
    type=str,
    choices=[
        "task_1_tumor_vs_normal",
        "task_2_tumor_subtyping",
        "wsi_classification",
        "wsi_classification_binary",
        "wsi_classification_binary_symptoms",
    ],
    default="wsi_classification_binary",
)
parser.add_argument(
    "--subtyping", action="store_true", default=True, help="subtyping problem"
)
parser.add_argument(
    "--bag_weight",
    type=float,
    default=1.0,
    help="clam: weight coefficient for bag-level loss (default: 0.7)",
)
parser.add_argument(
    "--tile_size",
    type=int,
    default=512,
    help="tile size (eg. 512x512)",
)
parser.add_argument(
    "--B",
    type=int,
    default=96,
    help="number of positive/negative patches to sample for clam",
)
parser.add_argument(
    "--drop_out", action="store_true", default=True, help="enabel dropout (p=0.25)"
)

parser.add_argument(
    "--save_img",
    action="store_true",
    default=True,
    help="whether to save heatmap images",
)
parser.add_argument("--h5_dir", type=str, help="directory of .h5 files fo each slide")
parser.add_argument(
    "--csv_in",
    type=str,
    help="path to csv file containing case_id, sample_id, filename, label",
)
parser.add_argument(
    "--csv_out", type=str, help="path to csv file where to save prediction scores"
)
parser.add_argument("--out_dir", type=str, help="directory of where to store images")
parser.add_argument(
    "--model_checkpoint", type=str, help="path to checkpoint of CLAM model"
)
args = parser.parse_args()

args.n_classes = 2

SAVE_IMAGES = args.save_img


def create_model(args):
    print("\nInit Model...", end=" ")
    model_dict = {"dropout": args.drop_out, "n_classes": args.n_classes}
    # Model type
    if args.model_type == "clam" and args.subtyping:
        model_dict.update({"subtyping": True})
    # Model size
    if args.model_size is not None and args.model_type != "mil":
        model_dict.update({"size_arg": args.model_size})
    # Model type 2: single branch, multiple branch
    if args.model_type in ["clam_sb", "clam_mb"]:
        if args.subtyping:
            model_dict.update({"subtyping": True})
        # B parameter for clustering
        if args.B > 0:
            model_dict.update({"k_sample": args.B})

        ## CREATION OF THE MODEL
        if args.model_type == "clam_sb":
            model = CLAM_SB(**model_dict)
        elif args.model_type == "clam_mb":
            model = CLAM_MB(**model_dict)
        else:
            raise NotImplementedError

    return model


def compute_iph(case_id, filename, out_df, gt_label):
    f = h5py.File(f"{args.h5_dir}{case_id}.h5")
    coords = f["coords"][:]
    features = f["features"][:]

    with torch.no_grad():
        # model returns logits, Y_prob, Y_hat, A_raw, results_dict
        if torch.cuda.is_available():
            res = model(torch.from_numpy(features).cuda())
        else:
            res = model(torch.from_numpy(features))

    # get patch wise scores and multiply them by bag shape
    patch_wise_scores = res[4]["patch_scores"] * features.shape[0]
    # apply sigmoid to identify positively contributing (s > 0.5) patches
    patch_wise_scores = torch.nn.Sigmoid()(patch_wise_scores)

    print(f"Probability of IPH: {round(res[1][0].item(), 4)}")

    area = (patch_wise_scores > 0.75).sum() / features.shape[0]
    print(f"Proxy IPH area: {area}")
    out_df = out_df.append(
        {
            "case_id": case_id,
            "IPH_pred": (res[1][0] > 0.5).detach().cpu().numpy(),
            "IPH_prob": (res[1][0]).detach().cpu().numpy(),
            "IPH_area": ((patch_wise_scores > 0.75).sum() / features.shape[0])
            .cpu()
            .numpy(),
            "ground_truth": gt_label == "yes",
            "bag_shape": features.shape[0]
        },
        ignore_index=True,
    )

    if SAVE_IMAGES:
        slide = openslide.open_slide(filename)
        # get downscaled version of the slide
        downscaled_img, _ = slide_to_scaled_pil_image(slide, SCALE_FACTOR=SCALE_FACTOR)
        downscaled_img.save(f"{args.out_dir}{case_id}_raw.png")
        # matrix to store logits
        logits = np.zeros((downscaled_img.size[1], downscaled_img.size[0]))
        # mask to store tissue locations
        mask = np.zeros((downscaled_img.size[1], downscaled_img.size[0]))

        # mask df will contain the iph/no iph binary information per patch; useful to overlay with cell
        # counts from stainings to define the cell type compositions of iph
        mask_df = pd.DataFrame(columns=["x", "y", "iph"])

        for coord, p in zip(coords, patch_wise_scores):
            y, x = map(int, coord)
            # mask_df = mask_df.append({'x':x, 'y':y, 'iph': int(p>0.5)}, ignore_index=True)
            mask_df = mask_df.append(
                {"x": x, "y": y, "iph": p.item()}, ignore_index=True
            )
            x //= SCALE_FACTOR
            y //= SCALE_FACTOR
            xspan, yspan = (
                slice(x, x + args.tile_size // SCALE_FACTOR),
                slice(y, y + args.tile_size // SCALE_FACTOR),
            )
            logits[xspan, yspan] += p.detach().cpu().numpy().item()
            mask[xspan, yspan] += 1

        logits /= mask + 1e-9
        # arr_min = np.min(logits[mask >= 1])
        # arr_max = np.max(logits[mask >= 1])
        arr_min = 0
        arr_max = 1
        arr = (logits - arr_min) / (arr_max - arr_min)
        # Convert the logits array into heatmap
        cmap = plt.get_cmap("coolwarm")
        rgba = cmap(arr, bytes=True)

        # Convert the 3D array to a PIL image
        img = Image.fromarray(rgba, "RGBA")
        mask = mask > 0
        alpha = 170 * mask.astype(np.uint8)
        img.putalpha(Image.fromarray(alpha, "L"))
        # Overlay heatmap to raw image
        overlaid_img = Image.alpha_composite(
            downscaled_img.convert("RGBA"), img
        ).convert("RGB")
        overlaid_img.save(f"{args.out_dir}{case_id}_overlay.png")
        mask_df.to_csv(f"{args.out_dir}{case_id}_mask.csv")
    else:
        # mask df will contain the iph/no iph binary information per patch; useful to overlay with cell
        # counts from stainings to define the cell type compositions of iph
        mask_df = pd.DataFrame(columns=["x", "y", "iph"])

        for coord, p in zip(coords, patch_wise_scores):
            y, x = map(int, coord)
            # mask_df = mask_df.append({'x':x, 'y':y, 'iph': int(p>0.5)}, ignore_index=True)
            mask_df = mask_df.append(
                {"x": x, "y": y, "iph": p.item()}, ignore_index=True
            )

        mask_df.to_csv(f"{args.out_dir}{case_id}_mask.csv")

    return out_df

model = initiate_model(args, args.model_checkpoint)
model.eval()

os.makedirs(args.out_dir, exist_ok=True)
out_df = pd.DataFrame()

df = pd.read_csv(args.csv_in)
# df = pd.read_csv('samples.txt', sep = ' ', header=None)
for idx, line in df.iterrows():
    case_id, sample_id, filename, gt_label = line
    # sample_id, filename = line
    print(sample_id)
    out_df = compute_iph(sample_id, filename, out_df, gt_label)
    
out_df.to_csv(args.csv_out)