import tensorflow as tf

MAX_PIXEL_VALUE = tf.constant(1.0)

def PSNR(y, p):
    """
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
    model.add(tf.keras.layers.Conv2D(input_shape=input_shape, filters=d, kernel_size=5, strides=(1, 1), padding="same",
                               data_format="channels_last"))
    model.add(tf.keras.layers.PReLU())
    """
    The second convolution is the Shrinking which is denoted Conv(1, s, d) in the paper.
    """
    model.add(tf.keras.layers.Conv2D(filters=s, kernel_size=1, strides=(1, 1), padding="same"))
    model.add(tf.keras.layers.PReLU())

    """
    The third part consists of m convolutional layers each denoted Conv(3, s, s) in the paper.
    """
    for i in range(m):
        model.add(tf.keras.layers.Conv2D(filters=s, kernel_size=3, strides=(1, 1), padding="same"))
        model.add(tf.keras.layers.PReLU())

    """
    The fourth part is the expanding part which is denoted Conv(1, d, s) in the paper. Note that this is the 
    opposite of the shrinking, that is in the shrinking part we go from channels 56 -> 12 and here we go from 12 -> 56.
    """
    model.add(tf.keras.layers.Conv2D(filters=d, kernel_size=1, strides=(1, 1), padding="same"))
    model.add(tf.keras.layers.PReLU())

    """
    The deconvolution part is denoted DeConv(9, 1, d) in the paper. The upscaling factor is decided by the stride.
    Note that here we use Conv2DTranspose: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose,
    This layer is sometimes called Deconvolution.
    """
    model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=9, strides=(upscaling, upscaling), padding="same"))

    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['mean_squared_error', PSNR])
    model.build()
    return model
#fsrcnn = FSRCNN(input_shape=(100, 100, 1), d=56, s=12, m=4, upscaling=3)
fsrcnn = FSRCNN(input_shape=(100, 100, 1), d=32, s=5, m=1, upscaling=3)
fsrcnn.summary()
#model.fit(x_train, y_train, epochs=5)
#model.evaluate(x_test, y_test)
