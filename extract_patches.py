from data import *
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-path", "--path", required=True, help="Path to the augmented images")
ap.add_argument("-output_path", "--output_path", required=True, help="Path to the output path for the data")
ap.add_argument("-f_sub_lr", "--f_sub_lr", required=True, type=int, help="Size of lr patches")
ap.add_argument("-aug", "--aug", required=False, action="store_true", help="Size of hr patches")
ap.add_argument("-upscaling", "--upscaling", required=True, type=int, help="Upscaling factor")
args = vars(ap.parse_args())

x, y = generate_2(dataset_folder=args['path'],
                 output_path=args['output_path'],
                 f_sub_lr=args['f_sub_lr'],
                 aug=args['aug'],
                 upscaling=args['upscaling'])
#np.savez(args['output_path'], x=x, y=y)
#python extract_patches.py -path "[augmented_data]" -output_path "" -f_sub_lr 7 -f_sub_hr 21 -stride 4 -upscaling 3
#mv data.npz data_T91_f_sub_lr=7_f_sub_hr=21_stride=4_upscaling=3.npz
#python extract_patches.py -path "[val_data]" -output_path "" -f_sub_lr 7 -f_sub_hr 21 -stride 4 -upscaling 3
#mv data.npz data_BSD500_20images_f_sub_lr=7_f_sub_hr=21_stride=4_upscaling=3.npz

