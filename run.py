from data import *
from fsrcnn import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-train_path", "--train_path", required=True, help="Path to the training data")
ap.add_argument("-val_path", "--val_path", required=True, help="Path to the validation data")
ap.add_argument("-batch_size", "--batch_size", required=True, type=int, help="Batch size during training")
ap.add_argument("-epochs", "--epochs", required=True, type=int, help="Number of epochs")
ap.add_argument("-f_sub_lr", "--f_sub_lr", required=True, type=int, help="Size of lr patches")
ap.add_argument("-f_sub_hr", "--f_sub_hr", required=True, type=int, help="Size of hr patches")
ap.add_argument("-stride", "--stride", required=True, type=int, help="Stride of patches")
ap.add_argument("-upscaling", "--upscaling", required=True, type=int, help="Upscaling factor")
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
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='tensorboard_logs'
)


fsrcnn = FSRCNN(input_shape=(f_sub_lr, f_sub_lr, 1),
                d=config[config_to_run][0],
                s=config[config_to_run][1],
                m=config[config_to_run][2],
                upscaling=upscaling)

fsrcnn.summary()
param_count = 0
for i in range(0, len(fsrcnn.layers), 2):
    param_count += fsrcnn.layers[i].count_params()
print("Number of parameters (PReLU not included):", param_count)

dat = np.load(data_path)
x = dat['x']
y = dat['y']

val_dat = np.load(val_data_path)
val_x = val_dat['x']
val_y = val_dat['y']#eh?

#fsrcnn.load_weights(checkpoint_path)
history = fsrcnn.fit(x=x,
           y=y,
           epochs=epochs,
           validation_data=(val_x, val_y),
           batch_size=batch_size,
           callbacks=[cp_callback, tensorboard_callback])