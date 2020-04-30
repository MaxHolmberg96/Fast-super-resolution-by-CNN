import tensorflow as tf


def FSRCNN(input_shape, d, s, m):
    """
    No channel depth is set for the filters, this is due to the filters
    channels automatically being set to the same number of channels as the
    input. For more information see: https://stackoverflow.com/a/45055094/13185722
    So it is called Conv2D because it is sliding across two dimensions only.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=input_shape, filters=d, kernel_size=5, strides=(1, 1), data_format="channels_last"),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters=s, kernel_size=1, strides=(1, 1)),
        tf.keras.layers.PReLU(),
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.build()
    return model
fsrcnn = FSRCNN(input_shape=(100, 100, 1), d=56, s=0, m=0)
fsrcnn.summary()
#model.fit(x_train, y_train, epochs=5)
#model.evaluate(x_test, y_test)
