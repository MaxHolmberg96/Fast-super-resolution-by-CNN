import tensorflow as tf
import pathlib
from tqdm import tqdm
import os
import numpy as np


def save_augmented_data(dataset, save_folder):
    data_dir = pathlib.Path(dataset)
    _, extension = os.path.splitext(os.listdir(data_dir)[3])
    for i in tqdm(data_dir.glob(f"*{extension}")):
        # Use grayscale because it is equivalent to first channel of yuv
        img = tf.keras.preprocessing.image.load_img(str(i), color_mode="grayscale")
        for scale in [1, 0.9, 0.8, 0.7, 0.6]:
            for rot in range(4):
                hr = tf.keras.preprocessing.image.img_to_array(img)
                hr = tf.image.rot90(hr, k=rot)
                h, w, _ = hr.shape
                hr = tf.image.resize(
                    tf.identity(hr),
                    (int(scale * h), int(scale * w)),
                    method=tf.image.ResizeMethod.BICUBIC,
                )
                if str(i).find("\\") == -1:
                    name = str(i).split("/")[2].split(".")[0]
                else:
                    name = str(i).split("\\")[2].split(".")[0]
                tf.keras.preprocessing.image.save_img(f"{save_folder}/{name}_rot={rot*90}_scale={scale}.{extension}", x=hr)


def generator(dataset_folder, f_sub_lr, f_sub_hr, k, upscaling):
    data_dir = pathlib.Path(dataset_folder)
    _, extension = os.path.splitext(os.listdir(data_dir)[3])
    paths = np.array(list(data_dir.glob(f"*{extension}")))
    x = []
    y = []
    for p in tqdm(paths):
        img = tf.keras.preprocessing.image.load_img(str(p), color_mode="grayscale")
        hr = tf.keras.preprocessing.image.img_to_array(img)
        h, w, _ = hr.shape
        new_w = int(w / upscaling)
        new_h = int(h / upscaling)
        lr = tf.image.resize(tf.identity(hr), (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC)
        lr_patches = tf.image.extract_patches(
            images=tf.expand_dims(lr, 0),
            sizes=[1, f_sub_lr, f_sub_lr, 1],
            strides=[1, k, k, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        hr_patches = tf.image.extract_patches(
            images=tf.expand_dims(hr, 0),
            sizes=[1, f_sub_hr, f_sub_hr, 1],
            strides=[1, k * upscaling, k * upscaling, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        size = 0
        for j in range(lr_patches.shape[1]): # Horizontal strides
            for l in range(lr_patches.shape[2]): # Vertical strides
                lr_patch = tf.reshape(lr_patches[0, j, l], (1, f_sub_lr, f_sub_lr, 1))
                hr_patch = tf.reshape(hr_patches[0, j, l], (1, f_sub_hr, f_sub_hr, 1))
                x.append(lr_patch)
                y.append(hr_patch)
    print("Let's concatenate")
    return tf.concat(x, 0) / tf.keras.backend.max(x), tf.concat(y, 0) / tf.keras.backend.max(y)


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
            image[i * f_sub: (i + 1) * f_sub, j * f_sub: (j + 1) * f_sub] = p_reshape[:, :, 0]
            count += 1
    return image