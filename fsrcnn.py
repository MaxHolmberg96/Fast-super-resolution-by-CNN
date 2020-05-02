"""
Packages:
    pip install tensorflow
    pip install Pillow

"""


import tensorflow as tf
import pathlib
from PIL import Image

MAX_PIXEL_VALUE = tf.constant(255.0)
general100_path = "C:/Users/Max/edu/kth/DD2424/project/dataset/General-100/"
image91_path = "C:/Users/Max"

def dataset_preparation(dataset, f_sub_lr, f_sub_hr, k, n):
    """
    0. Read in all images in a 4d tensor of shape [batch, size1, size2, channels]

    1. Convert images to illuminance with: tf.image.rgb_to_yuv and extract first channel

    2. Downscale images by a factor of n (upscale factor)
        tf.image.resize(
            images, size, method=ResizeMethod.BICUBIC, preserve_aspect_ratio=False,
            antialias=False, name=None
        )


    3. We're gonna extract patches of size (f_sub x f_sub) with stride k, following:
        https://www.tensorflow.org/api_docs/python/tf/image/extract_patches we can do this easily.


    :param dataset: The dataset to extract patches and prepare
    """
    data_dir = pathlib.Path(dataset)
    for i in data_dir.glob('*.bmp'):
        # Use grayscale because it is equivalent to first channel of yuv
        img = tf.keras.preprocessing.image.load_img(str(i), color_mode='grayscale')
        w, h = img.size
        hr = tf.keras.preprocessing.image.img_to_array(img)
        new_w = int(w / n)
        new_h = int(h / n)
        lr = tf.image.resize(tf.identity(hr), (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC)
        lr_patches = tf.image.extract_patches(images=tf.expand_dims(lr, 0),
                                              sizes=[1, f_sub_lr, f_sub_lr, 1],
                                              strides=[1, k, k, 1],
                                              rates=[1, 1, 1, 1],
                                              padding='VALID')
        hr_patches = tf.image.extract_patches(images=tf.expand_dims(hr, 0),
                                              sizes=[1, f_sub_hr, f_sub_hr, 1],
                                              strides=[1, k, k, 1],
                                              rates=[1, 1, 1, 1],
                                              padding='VALID')
        for j in range(lr_patches.shape[1]):
            for l in range(lr_patches.shape[2]):
                lr_patch = tf.reshape(lr_patches[0, j, l], (1, f_sub_lr, f_sub_lr, 1))
                hr_patch = tf.reshape(hr_patches[0, j, l], (1, f_sub_hr, f_sub_hr, 1))
                yield (lr_patch, hr_patch)


def psnr(y, p):
    """
    Implemented from the wiki page of PSNR.

    :param y: target value.
    :param p: predicted.
    :param max_pixel_value: The max pixel value we're using.

    :return: Peak signal-to-noise ratio
    """
    def log10(x):
        return tf.math.log(x) / tf.math.log(10.0)
    return 10 * log10(tf.math.pow(MAX_PIXEL_VALUE, 2) / tf.keras.losses.MSE(y_true=y, y_pred=p))


def FSRCNN(input_shape, d, s, m, upscaling):
    """
    No channel depth is set for the filters, this is due to the filters
    channels automatically being set to the same number of channels as the
    input. For more information see: https://stackoverflow.com/a/45055094/13185722
    So it is called Conv2D because it is sliding across two dimensions only.
    """
    model = tf.keras.models.Sequential()
    """
    The first convolution is the Feature Extraction which is denoted Conv(5, d, 1) in the paper. Channels = 1 is 
    automatically set to one because the input has 1 channel.
    """
    model.add(tf.keras.layers.Conv2D(input_shape=input_shape, filters=d, kernel_size=5, padding="same",
                               data_format="channels_last", use_bias=False))
    model.add(tf.keras.layers.PReLU())
    """
    The second convolution is the Shrinking which is denoted Conv(1, s, d) in the paper.
    """
    model.add(tf.keras.layers.Conv2D(filters=s, kernel_size=1, padding="same", use_bias=False))
    model.add(tf.keras.layers.PReLU())

    """
    The third part consists of m convolutional layers each denoted Conv(3, s, s) in the paper.
    """
    for i in range(m):
        model.add(tf.keras.layers.Conv2D(filters=s, kernel_size=3, padding="same", use_bias=False))
        model.add(tf.keras.layers.PReLU())

    """
    The fourth part is the expanding part which is denoted Conv(1, d, s) in the paper. Note that this is the 
    opposite of the shrinking, that is in the shrinking part we go from channels 56 -> 12 and here we go from 12 -> 56.
    """
    model.add(tf.keras.layers.Conv2D(filters=d, kernel_size=1, padding="same", use_bias=False))
    model.add(tf.keras.layers.PReLU())

    """
    The deconvolution part is denoted DeConv(9, 1, d) in the paper. The upscaling factor is decided by the stride.
    Note that here we use Conv2DTranspose: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose,
    This layer is sometimes called Deconvolution.
    """
    model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=9, strides=(upscaling, upscaling), padding="same", use_bias=False))
    sgd = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.0)
    model.compile(optimizer=sgd,
                  loss='mean_squared_error',
                  metrics=['mean_squared_error', psnr])
    model.build()
    return model
upscaling = 2
fsrcnn = FSRCNN(input_shape=(100, 100, 1), d=56, s=12, m=4, upscaling=upscaling)
#fsrcnn = FSRCNN(input_shape=(f_sub, f_sub, 1), d=32, s=5, m=1, upscaling=upscaling)

fsrcnn.summary()
param_count = 0
for i in range(0, len(fsrcnn.layers), 2):
    param_count += fsrcnn.layers[i].count_params()
print("Number of parameters (PReLU not included):", param_count)


#Upscaling factor: 2x = f_sub_lr=10, f_sub_hr=10
#Upscaling factor: 3x = f_sub_lr=7, f_sub_hr=19
#Upscaling factor: 4x = f_sub_lr=6, f_sub_hr=21
fsrcnn.fit(dataset_preparation(general100_path, f_sub_lr=10, f_sub_hr=19, k=4, n=upscaling), epochs=5)
