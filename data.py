import tensorflow as tf
import pathlib
from tqdm import tqdm
import os
import numpy as np
import cv2


def save_augmented_data(dataset, save_folder):
    data_dir = pathlib.Path(dataset)
    _, extension = os.path.splitext(os.listdir(data_dir)[3])
    for i in tqdm(data_dir.glob(f"*{extension}")):
        # Use grayscale because it is equivalent to first channel of yuv
        img = cv2.imread(str(i)) #cv2 uses bgr as default: https://stackoverflow.com/a/39316695
        ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = tf.expand_dims(ycrcb_image[:, :, 0], 2)
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
                tf.keras.preprocessing.image.save_img(f"{save_folder}/{name}_rot={rot*90}_scale={scale}{extension}", x=hr)


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
                lr_patch = tf.reshape(lr_patches[0, j, l], (1, f_sub_lr, f_sub_lr, 1)).numpy()
                hr_patch = tf.reshape(hr_patches[0, j, l], (1, f_sub_hr, f_sub_hr, 1)).numpy()
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


def evaluate_on_dataset(fsrcnn, dataset, upscaling, should_extract_patches=False, f_sub_lr=0, f_sub_hr=0, k=0):
    psnr = []
    if should_extract_patches:
        x, y = generator(dataset, f_sub_lr, f_sub_hr, k, upscaling)
        res = fsrcnn.evaluate(x, y)
        psnr.append(res[1])
        return tf.reduce_mean(psnr)
    data_dir = pathlib.Path(dataset)
    _, extension = os.path.splitext(os.listdir(data_dir)[3])
    for i in tqdm(data_dir.glob(f"*{extension}")):
        img = tf.keras.preprocessing.image.load_img(str(i), color_mode="grayscale")
        hr = tf.keras.preprocessing.image.img_to_array(img)
        hr = modcrop(hr, scale=upscaling)
        h, w, _ = hr.shape
        new_w = int(w / upscaling)
        new_h = int(h / upscaling)
        lr = tf.image.resize(tf.identity(hr), (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC)
        lr = tf.expand_dims(lr, 0)
        hr = tf.expand_dims(hr, 0)
        y = hr / np.max(hr)
        x = lr / np.max(lr)
        res = fsrcnn.evaluate(x, y)
        psnr.append(res[1])
    return tf.reduce_mean(psnr)


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
        x.append(tf.expand_dims(hr, 0))


    pickle.dump(x)
    pickle.dump(y)


#create_pickle_from_folder("dataset/Set5", "Set5.npz", 3)
#create_pickle_from_folder("dataset/Set14", "Set14.npz", 3)
#create_pickle_from_folder("dataset/BSD200", "BSD200.npz", 3)
