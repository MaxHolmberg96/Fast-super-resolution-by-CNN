import tensorflow as tf
import pathlib
from tqdm import tqdm
import os
import numpy as np
import cv2
import pickle


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
                hr = tf.keras.preprocessing.image.img_to_array(img, dtype=tf.float64)
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
        img = cv2.imread(str(p))  # cv2 uses bgr as default: https://stackoverflow.com/a/39316695
        ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        hr = tf.expand_dims(ycrcb_image[:, :, 0], 2)
        #hr = tf.keras.preprocessing.image.img_to_array(img)
        h, w, _ = hr.shape
        # Resize hr again to make sure it is exactly upscaled by 3 and not > 3
        hr = tf.image.resize(
            tf.identity(hr),
            ((h // upscaling) * upscaling, (w // upscaling) * upscaling),
            method=tf.image.ResizeMethod.BICUBIC,
        )
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

def generate_2(dataset_folder, output_path, f_sub_lr, aug, upscaling):
    import h5py
    import glob
    from PIL import Image
    h5_file = h5py.File(output_path, 'w')
    lr_patches = []
    hr_patches = []

    for image_path in tqdm(sorted(glob.glob('{}/*'.format(dataset_folder)))):
        hr = Image.open(image_path).convert('RGB')
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
                    lr_patches.append(lr[i:i+f_sub_lr, j:j+f_sub_lr])
                    hr_patches.append(hr[i*upscaling:i*upscaling+f_sub_lr*upscaling, j*upscaling:j*upscaling+f_sub_lr*upscaling])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


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
        x.append(tf.expand_dims(lr, 0) / tf.keras.backend.max(lr))
        y.append(tf.expand_dims(hr, 0) / tf.keras.backend.max(lr))



    pickle.dump({'x': x, 'y': y}, open(save_folder, "wb"))


# save_augmented_data("dataset/Set14", "dataset/Set14-aug")
# create_pickle_from_folder("dataset/Set5", "../Set5.p", 3)
# create_pickle_from_folder("dataset/Set14", "../Set14.p", 3)
# create_pickle_from_folder("dataset/BSD200", "../BSD200.p", 3)
