"""
Packages:
    pip install tensorflow
    pip install Pillow
    pip install tqdm
    pip install -q pyyaml h5py
"""
import tensorflow as tf

def psnr(y_pred, y_true, clip=True):
    ps = tf.image.psnr(y_true, y_pred, max_val=1.0)
    if clip:
        return tf.clip_by_value(ps, clip_value_min=0, clip_value_max=99.9)
    return ps


def FSRCNN(d, s, m, upscaling):
    """
    No channel depth is set for the filters, this is due to the filters
    channels automatically being set to the same number of channels as the
    input. For more information see: https://stackoverflow.com/a/45055094/13185722
    So it is called Conv2D because it is sliding across two dimensions only.

    The initialization of all the layers except the DeConv follows: https://arxiv.org/pdf/1502.01852.pdf which is
    the He initializer.
    """
    model = tf.keras.models.Sequential()
    bias = True
    """
    The first convolution is the Feature Extraction which is denoted Conv(5, d, 1) in the paper. Channels = 1 is
    automatically set to one because the input has 1 channel.
    """
    model.add(
        tf.keras.layers.Conv2D(
            input_shape=(None, None, 1),
            filters=d,
            kernel_size=5,
            padding="same",
            data_format="channels_last",
            use_bias=bias,
            kernel_initializer=tf.keras.initializers.he_normal(),
        )
    )
    model.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))
    """
    The second convolution is the Shrinking which is denoted Conv(1, s, d) in the paper.
    """
    model.add(
        tf.keras.layers.Conv2D(
            filters=s,
            kernel_size=1,
            padding="same",
            use_bias=bias,
            kernel_initializer=tf.keras.initializers.he_normal(),
        )
    )
    model.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))

    """
    The third part consists of m convolutional layers each denoted Conv(3, s, s) in the paper.
    """
    for i in range(m):
        model.add(
            tf.keras.layers.Conv2D(
                filters=s,
                kernel_size=3,
                padding="same",
                use_bias=bias,
                kernel_initializer=tf.keras.initializers.he_normal(),
            )
        )
        model.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))

    """
    The fourth part is the expanding part which is denoted Conv(1, d, s) in the paper. Note that this is the
    opposite of the shrinking, that is in the shrinking part we go from channels 56 -> 12 and here we go from 12 -> 56.
    """
    model.add(
        tf.keras.layers.Conv2D(
            filters=d,
            kernel_size=1,
            padding="same",
            use_bias=bias,
            kernel_initializer=tf.keras.initializers.he_normal(),
        )
    )
    model.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))

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
            use_bias=bias,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
        )
    )
    model.build()

    return model
