import tensorflow as tf
import pathlib
from tqdm import tqdm
import os
import numpy as np
import cv2
import pickle

"""
Code for pre-processing for this project is heavily adapted from https://github.com/yjn870/FSRCNN-pytorch
"""

def create_patches(dataset_folder, output_path, f_sub_lr, aug, upscaling):
    import h5py
    import glob
    from PIL import Image

    h5_file = h5py.File(output_path, "w")
    lr_patches = []
    hr_patches = []

    for image_path in tqdm(sorted(glob.glob("{}/*".format(dataset_folder)))):
        hr = Image.open(image_path).convert("RGB")
        hr_images = []
        if aug:
            for s in [1.0, 0.9, 0.8, 0.7, 0.6]:
                for r in [0, 90, 180, 270]:
                    tmp = hr.resize((int(hr.width * s), int(hr.height * s)), resample=Image.BICUBIC)
                    tmp = tmp.rotate(r, expand=True)
                    hr_images.append(tmp)
        else:
            hr_images.append(hr)

        for hr in hr_images:
            hr_width = (hr.width // upscaling) * upscaling
            hr_height = (hr.height // upscaling) * upscaling
            hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
            lr = hr.resize((hr.width // upscaling, hr_height // upscaling), resample=Image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            for i in range(0, lr.shape[0] - f_sub_lr + 1, upscaling):
                for j in range(0, lr.shape[1] - f_sub_lr + 1, upscaling):
                    lr_patches.append(lr[i : i + f_sub_lr, j : j + f_sub_lr])
                    hr_patches.append(
                        hr[
                            i * upscaling : i * upscaling + f_sub_lr * upscaling,
                            j * upscaling : j * upscaling + f_sub_lr * upscaling,
                        ]
                    )

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset("lr", data=lr_patches)
    h5_file.create_dataset("hr", data=hr_patches)

    h5_file.close()


def convert_rgb_to_y(img, dim_order="hwc"):
    if dim_order == "hwc":
        return 16.0 + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.0
    else:
        return 16.0 + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.0


def convert_rgb_to_ycbcr(img, dim_order="hwc"):
    if dim_order == "hwc":
        y = 16.0 + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.0
        cb = 128.0 + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.0
        cr = 128.0 + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.0
    else:
        y = 16.0 + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.0
        cb = 128.0 + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.0
        cr = 128.0 + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.0
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def convert_ycbcr_to_rgb(img, dim_order="hwc"):
    if dim_order == "hwc":
        r = 298.082 * img[..., 0] / 256.0 + 408.583 * img[..., 2] / 256.0 - 222.921
        g = 298.082 * img[..., 0] / 256.0 - 100.291 * img[..., 1] / 256.0 - 208.120 * img[..., 2] / 256.0 + 135.576
        b = 298.082 * img[..., 0] / 256.0 + 516.412 * img[..., 1] / 256.0 - 276.836
    else:
        r = 298.082 * img[0] / 256.0 + 408.583 * img[2] / 256.0 - 222.921
        g = 298.082 * img[0] / 256.0 - 100.291 * img[1] / 256.0 - 208.120 * img[2] / 256.0 + 135.576
        b = 298.082 * img[0] / 256.0 + 516.412 * img[1] / 256.0 - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


def extract_patches(img, f_sub):
    patches = tf.image.extract_patches(
        images=tf.expand_dims(img, 0),
        sizes=[1, f_sub, f_sub, 1],
        strides=[1, f_sub, f_sub, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    x = []
    for j in range(patches.shape[1]):  # Horizontal strides
        for l in range(patches.shape[2]):  # Vertical strides
            x.append(tf.reshape(patches[0, j, l], (1, f_sub, f_sub, 1)))
    return tf.concat(x, 0) / tf.keras.backend.max(x), (patches.shape[1], patches.shape[2])


def put_togeheter_patches(patches, patches_shape, f_sub):
    image = np.zeros((patches_shape[0] * f_sub, patches_shape[1] * f_sub))
    count = 0
    for i in range(patches_shape[0]):
        for j in range(patches_shape[1]):
            p_reshape = patches[count]
            image[i * f_sub : (i + 1) * f_sub, j * f_sub : (j + 1) * f_sub] = p_reshape[:, :, 0]
            count += 1
    return image


def modcrop(image, scale=3):
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def create_pickle_from_folder(dataset, save_folder, upscaling):
    data_dir = pathlib.Path(dataset)
    _, extension = os.path.splitext(os.listdir(data_dir)[3])
    paths = np.array(list(data_dir.glob(f"*{extension}")))
    x = []
    y = []
    for path in tqdm(paths):
        img = cv2.imread(str(path))  # cv2 uses bgr as default: https://stackoverflow.com/a/39316695
        ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = tf.expand_dims(ycrcb_image[:, :, 0], 2)
        hr = tf.keras.preprocessing.image.img_to_array(img)
        hr = modcrop(hr, scale=3)
        h, w, _ = hr.shape
        new_w = int(w / upscaling)
        new_h = int(h / upscaling)
        lr = tf.image.resize(tf.identity(hr), (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC)
        x.append(tf.expand_dims(lr, 0) / tf.keras.backend.max(lr))
        y.append(tf.expand_dims(hr, 0) / tf.keras.backend.max(lr))

    pickle.dump({"x": x, "y": y}, open(save_folder, "wb"))