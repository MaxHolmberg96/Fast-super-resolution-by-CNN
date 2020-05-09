from data import *
from fsrcnn import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-train_path", "--train_path", required=True, help="Path to the training data")
ap.add_argument("-val_path", "--val_path", required=True, help="Path to the validation data")
ap.add_argument("-batch_size", "--batch_size", required=False, type=int, default=64, help="Batch size during training")
ap.add_argument("-epochs", "--epochs", required=False, type=int, default=100, help="Number of epochs")
ap.add_argument("-f_sub_lr", "--f_sub_lr", required=False, type=int, default=7, help="Size of lr patches")
ap.add_argument("-f_sub_hr", "--f_sub_hr", required=False, type=int, default=21, help="Size of hr patches")
ap.add_argument("-stride", "--stride", required=False, type=int, default=4, help="Stride of patches")
ap.add_argument("-upscaling", "--upscaling", required=False, type=int, default=3, help="Upscaling factor")
ap.add_argument("-init_epoch", "--init_epoch", required=False, type=int, default=0, help="Start epoch step for Tensorboard, useful for saving and loading")
args = vars(ap.parse_args())

"""
Paths
"""
checkpoint_path = "training_checkpoints/cp.ckpt"
data_path = args['train_path']
val_data_path = args['val_path']

"""
Hyperparameters for the model
"""
init_epoch = args['init_epoch']
upscaling = args['upscaling']
f_sub_lr = args['f_sub_lr']
f_sub_hr = args['f_sub_hr']
batch_size = args['batch_size']
epochs = args['epochs']
config = [
    (56, 12, 4),
    (32, 5, 1)
]
config_to_run = 0
"""
Create checkpoint callback
"""
class TensorBoard_callback(tf.keras.callbacks.TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        super_res = self.model.predict(lr)
        with train_summary_writer.as_default():
            tf.summary.image('Superres', super_res, step=epoch)


checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

tensorboard_callback = TensorBoard_callback(
    log_dir='tensorboard_logs',
    histogram_freq=5
)

"""
Should load more images with a function from the validation set which we can show in TensorBoard
"""
train_summary_writer = tf.summary.create_file_writer('tensorboard_logs/images/')
img = tf.keras.preprocessing.image.load_img('./dataset/Set5/baby_GT.bmp', color_mode="grayscale")
hr = tf.keras.preprocessing.image.img_to_array(img)
h, w, _ = hr.shape
lr = tf.expand_dims(
    tf.image.resize(
        tf.identity(hr), (h // upscaling, w // upscaling), method=tf.image.ResizeMethod.BICUBIC
    ),
    0,
)

# Making the model
fsrcnn = FSRCNN(input_shape=(f_sub_lr, f_sub_lr, 1),
                d=config[config_to_run][0],
                s=config[config_to_run][1],
                m=config[config_to_run][2],
                upscaling=upscaling,
                )

fsrcnn.summary()
param_count = 0
for i in range(0, len(fsrcnn.layers), 2):
    param_count += fsrcnn.layers[i].count_params()
print("Number of parameters (PReLU not included):", param_count)

# Loading training and validation data
dat = np.load(data_path)
x = dat['x']
y = dat['y']

val_dat = np.load(val_data_path)
val_x = val_dat['x']
val_y = val_dat['y']

fsrcnn.load_weights(checkpoint_path)
history = fsrcnn.fit(x=x,
                     y=y,
                     epochs=epochs,
                     validation_data=(val_x, val_y),
                     batch_size=batch_size,
                     callbacks=[cp_callback, tensorboard_callback],
                     initial_epoch=init_epoch
                     )
