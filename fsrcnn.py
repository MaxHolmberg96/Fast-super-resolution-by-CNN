"""
Packages:
    pip install tensorflow
    pip install Pillow

"""


import tensorflow as tf
import pathlib
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

MAX_PIXEL_VALUE = tf.constant(1.0)
general100_path = "dataset/General-100/"
image91_path = "dataset/T91/"


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
                name = str(i).split("\\")[2].split(".")[0]
                tf.keras.preprocessing.image.save_img(f"{save_folder}/{name}_rot={rot*90}_scale={scale}.{extension}", x=hr)


def generator(dataset_folder, batch_size, f_sub_lr, f_sub_hr, k, upscaling):
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
                #if size == batch_size:
                    #yield tf.concat(x, 0) / tf.keras.backend.max(x), tf.concat(y, 0) / tf.keras.backend.max(y)
                #    x.clear()
                #    y.clear()
                #    size = 0
                #else:
                x.append(lr_patch)
                y.append(hr_patch)
                #    size += 1
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


def psnr(y, p):
    ps = tf.image.psnr(y, p, max_val=1.0)
    #ps[tf.math.is_inf(ps)] = 0
    return ps


def FSRCNN(input_shape, d, s, m, upscaling):
    """
    No channel depth is set for the filters, this is due to the filters
    channels automatically being set to the same number of channels as the
    input. For more information see: https://stackoverflow.com/a/45055094/13185722
    So it is called Conv2D because it is sliding across two dimensions only.

    The initialization of all the layers except the DeConv follows: https://arxiv.org/pdf/1502.01852.pdf which is
    the He initializer.
    """
    model = tf.keras.models.Sequential()
    """
    The first convolution is the Feature Extraction which is denoted Conv(5, d, 1) in the paper. Channels = 1 is
    automatically set to one because the input has 1 channel.
    """
    model.add(
        tf.keras.layers.Conv2D(
            input_shape=input_shape,
            filters=d,
            kernel_size=5,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal(),
        )
    )
    model.add(tf.keras.layers.PReLU())
    """
    The second convolution is the Shrinking which is denoted Conv(1, s, d) in the paper.
    """
    model.add(
        tf.keras.layers.Conv2D(
            filters=s,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal(),
        )
    )
    model.add(tf.keras.layers.PReLU())

    """
    The third part consists of m convolutional layers each denoted Conv(3, s, s) in the paper.
    """
    for i in range(m):
        model.add(
            tf.keras.layers.Conv2D(
                filters=s,
                kernel_size=3,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.he_normal(),
            )
        )
        model.add(tf.keras.layers.PReLU())

    """
    The fourth part is the expanding part which is denoted Conv(1, d, s) in the paper. Note that this is the
    opposite of the shrinking, that is in the shrinking part we go from channels 56 -> 12 and here we go from 12 -> 56.
    """
    model.add(
        tf.keras.layers.Conv2D(
            filters=d,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.he_normal(),
        )
    )
    model.add(tf.keras.layers.PReLU())

    """
    The deconvolution part is denoted DeConv(9, 1, d) in the paper. The upscaling factor is decided by the stride.
    Note that here we use Conv2DTranspose: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose,
    This layer is sometimes called Deconvolution.
    """
    model.add(
        tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=9,
            strides=(upscaling, upscaling),
            padding="same",
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
        )
    )

    sgd = tf.keras.optimizers.Adam()
    model.compile(optimizer=sgd, loss="mean_squared_error", metrics=[psnr])
    model.build()
    return model


upscaling = 3
f_sub_lr = 7
f_sub_hr = f_sub_lr * upscaling
patch_stride = 4
fsrcnn = FSRCNN(input_shape=(f_sub_lr, f_sub_lr, 1), d=56, s=12, m=4, upscaling=upscaling)
# fsrcnn = FSRCNN(input_shape=(f_sub_lr, f_sub_lr, 1), d=32, s=5, m=1, upscaling=upscaling)

fsrcnn.summary()
param_count = 0
for i in range(0, len(fsrcnn.layers), 2):
    param_count += fsrcnn.layers[i].count_params()
print("Number of parameters (PReLU not included):", param_count)

# This is not used I think, as we use here f_sub_lr = a and then f_sub_hr = a * upscaling
# Upscaling factor: 2x = f_sub_lr=10, f_sub_hr=19
# Upscaling factor: 3x = f_sub_lr=7, f_sub_hr=19
# Upscaling factor: 4x = f_sub_lr=6, f_sub_hr=21

#save_augmented_data(image91_path, save_folder="dataset/T91-aug")
#x, y = generator("dataset/General-100-aug/", batch_size=1, f_sub_lr=f_sub_lr, f_sub_hr=f_sub_hr, k=patch_stride, upscaling=upscaling)
#np.savez("data", x=x, y=y)

dat = np.load("data.npz")
x = dat['x']
y = dat['y']
indices = np.random.choice(np.arange(x.shape[0]), 70000)
val_x = x[indices]
val_y = y[indices]
x = np.delete(x, indices, 0)
y = np.delete(y, indices, 0)


fsrcnn.fit(x, y, epochs=10, batch_size=32, validation_data=(val_x, val_y))
fsrcnn.evaluate(val_x, val_y)


img = tf.keras.preprocessing.image.load_img("dataset/General-100/im_8.bmp", color_mode="grayscale")
hr = tf.keras.preprocessing.image.img_to_array(img)
h, w, _ = hr.shape
new_w = int(w / upscaling)
new_h = int(h / upscaling)
lr = tf.image.resize(tf.identity(hr), (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC)
print(lr.shape)
tf.keras.preprocessing.image.save_img(path="lr.bmp", x=lr)
patches, patches_shape = extract_patches(lr, f_sub_lr)
#print(patches.shape)

patches_pred = fsrcnn.predict(patches)
image = put_togeheter_patches(patches_pred, patches_shape, f_sub_hr)
print(image.shape)
tf.keras.preprocessing.image.save_img(path="upscaled_lr.bmp", x=tf.expand_dims(image, 2))