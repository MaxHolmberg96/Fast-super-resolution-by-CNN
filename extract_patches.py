import argparse

from data import create_patches

ap = argparse.ArgumentParser()
ap.add_argument("-path", "--path", required=True, help="Path to the augmented images")
ap.add_argument("-output_path", "--output_path", required=True, help="Path to the output path for the data")
ap.add_argument("-f_sub_lr", "--f_sub_lr", required=True, type=int, help="Size of lr patches")
ap.add_argument("-aug", "--aug", required=False, action="store_true", help="Size of hr patches")
ap.add_argument("-upscaling", "--upscaling", required=True, type=int, help="Upscaling factor")
args = vars(ap.parse_args())

create_patches(
    dataset_folder=args["path"],
    output_path=args["output_path"],
    f_sub_lr=args["f_sub_lr"],
    aug=args["aug"],
    upscaling=args["upscaling"],
)