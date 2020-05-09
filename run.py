from data import *
from fsrcnn import *
import datetime
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
class custom_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(logs.keys())
        with train_summary_writer.as_default():
            tf.summary.scalar("MSE", logs['loss'], step=epoch)
            tf.summary.scalar("VAL_MSE", logs['val_loss'], step=epoch)
            tf.summary.scalar("PSNR", logs['psnr'], step=epoch)
            tf.summary.scalar("VAL_PSNR", logs['val_psnr'], step=epoch)
        for i, show_image in enumerate(show_images):
            pred = self.model.predict(show_image)
            with image_summary_writer.as_default():
                tf.summary.image(str(i), tf.concat(hr_and_bicubic[i] + [pred], 0), step=epoch)



checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

image_summary_writer = tf.summary.create_file_writer('tensorboard_logs/images/')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'tensorboard_logs/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
"""
Should load more images with a function from the validation set which we can show in TensorBoard
"""
data_dir = pathlib.Path("dataset/Set5/")
hr_and_bicubic = []
show_images = []
_, extension = os.path.splitext(os.listdir(data_dir)[3])
for file in tqdm(data_dir.glob(f"*{extension}")):
    img = tf.keras.preprocessing.image.load_img(str(file), color_mode="grayscale")
    hr = tf.keras.preprocessing.image.img_to_array(img)
    h, w, _ = hr.shape
    lr = tf.expand_dims(
        tf.image.resize(
            tf.identity(hr), (h // upscaling, w // upscaling), method=tf.image.ResizeMethod.BICUBIC
        ),
        0,
    )
    hr /= np.max(hr)
    hr = tf.expand_dims(modcrop(hr, upscaling), 0)
    lr /= np.max(lr)
    show_images.append(lr)
    bicubic = tf.image.resize(lr[0], (lr.shape[1] * upscaling, lr.shape[2] * upscaling), method=tf.image.ResizeMethod.BICUBIC)
    hr_and_bicubic.append([hr, tf.expand_dims(bicubic, 0)])


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

history = fsrcnn.fit(x=x,
                     y=y,
                     epochs=epochs,
                     validation_data=(val_x, val_y),
                     batch_size=batch_size,
                     callbacks=[cp_callback, custom_callback()],
                     initial_epoch=init_epoch
)
