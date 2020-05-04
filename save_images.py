from data import *
import argparse

ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-path", "--path", required=True, help="Path to the augmented images")
ap.add_argument("-output_path", "--output_path", required=True, help="Path to the output path for the data")
args = vars(ap.parse_args())
save_augmented_data(args['path'], save_folder=args['output_path'])