from data import *
from fsrcnn import *
import datetime
import argparse
from custom_adam import *


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
hyperparams = {
    'adam_alpha': 1e-3,
    'update_losses': 1000,
    'weights_folder': 'weights_folder/',
    'logs_images_folder': 'tensorboard_logs/' + current_time + '_images/',
    'logs_scalar_folder': 'tensorboard_logs/' + current_time + '/train',
}


@tf.function
def train_step(x, y):
    with tf.GradientTape() as g:
        mse = fsrcnn_loss(fsrcnn, x, y)
    gradients = g.gradient(mse, fsrcnn.trainable_variables)
    fsrcnn_optimizer.apply_gradients(zip(gradients, fsrcnn.trainable_variables))
    return gradients

@tf.function
def fsrcnn_loss(model, x, y_true):
    mse = tf.keras.losses.MeanSquaredError()
    y_pred = model(x)
    return mse(y_true, y_pred)

@tf.function
def PSNR(model, x, y_true):
    y_pred = model(x)
    ps = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return tf.clip_by_value(ps, clip_value_min=0, clip_value_max=99.9)


def train(x, y, val_x, val_y, epochs, ckpt_manager, shuffle=True, initial_log_step=0):
    import time
    from tqdm import tqdm

    if shuffle:
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

    if args['continue']:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        if ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
    else:
        print("Initializing from scratch.")
    update_step = 0
    update_log_step = initial_log_step
    for epoch in range(epochs):
        start = time.time()
        offset = 0
        iterator = tqdm(range(x.shape[0] // hyperparams['batch_size']))
        for i in iterator:
            batch_x = x[offset:offset + hyperparams['batch_size']]
            batch_y = y[offset:offset + hyperparams['batch_size']]
            grads = train_step(batch_x, batch_y)
            if update_step % hyperparams['update_losses'] == 0:
                loss = fsrcnn_loss(fsrcnn, batch_x, batch_y)
                val_loss = fsrcnn_loss(fsrcnn, val_x, val_y)
                psnr = np.mean(PSNR(fsrcnn, batch_x, batch_y))
                val_psnr = np.mean(PSNR(fsrcnn, val_x, val_y))
                iterator.set_description("\nloss: {:.5f}, val_loss: {:.5f}, psnr: {:.5f}, val_psnr: {:.5f}".format(loss, val_loss, psnr, val_psnr))
                # Write to tensorboard
                write_batch_summaries(loss, val_loss, psnr, val_psnr, update_log_step)
                update_log_step += 1

            offset += hyperparams['batch_size']
            update_step += 1

        ckpt.step.assign_add(1)
        save_path = ckpt_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        write_epoch_summaries(grads, fsrcnn, epoch)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


def write_epoch_summaries(grads, model, epoch):
    for i, show_image in enumerate(show_images):
        pred = model.predict(show_image)
        with image_summary_writer.as_default():
            tf.summary.image(str(i), tf.concat(hr_and_bicubic[i] + [pred], 0), step=epoch)

    names = [model.layers[i].name for i in range(len(model.layers))]
    with train_summary_writer.as_default():
        for i in range(len(grads)):
            if i > len(names) - 1:
                name = "bias"
            else:
                name = names[i]
            tf.summary.histogram("grads_" + name, grads[i], step=epoch)
            tf.summary.scalar("grads_norm_" + name, tf.norm(grads[i], ord=2), step=epoch)
        weights = model.get_weights()
        for i in range(len(weights)):
            if i > len(names) - 1:
                name = "bias"
            else:
                name = names[i]
            tf.summary.histogram("weights_" + name, weights[i], step=epoch)

def write_batch_summaries(loss, val_loss, psnr, val_psnr, update_log_step):
    with train_summary_writer.as_default():
        tf.summary.scalar('mse', loss, step=update_log_step)
        tf.summary.scalar('val_mse', val_loss, step=update_log_step)
        tf.summary.scalar('psnr', psnr, step=update_log_step)
        tf.summary.scalar('val_psnr', val_psnr, step=update_log_step)

ap = argparse.ArgumentParser()
ap.add_argument("-train_path", "--train_path", required=True, help="Path to the training data")
ap.add_argument("-val_path", "--val_path", required=True, help="Path to the validation data")
ap.add_argument("-batch_size", "--batch_size", required=False, type=int, default=64, help="Batch size during training")
ap.add_argument("-epochs", "--epochs", required=False, type=int, default=100, help="Number of epochs")
ap.add_argument("-f_sub_lr", "--f_sub_lr", required=False, type=int, default=7, help="Size of lr patches")
ap.add_argument("-f_sub_hr", "--f_sub_hr", required=False, type=int, default=21, help="Size of hr patches")
ap.add_argument("-upscaling", "--upscaling", required=False, type=int, default=3, help="Upscaling factor")
ap.add_argument("-initial_log_step", "--initial_log_step", required=False, type=int, default=0, help="Where to start the log step")
ap.add_argument("-continue", "--continue", required=False, action="store_true", help="If we want to continue training")
args = vars(ap.parse_args())

"""
Parameters
"""
data_path = args['train_path']
val_data_path = args['val_path']
hyperparams['batch_size'] = args['batch_size']
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
image_summary_writer = tf.summary.create_file_writer(hyperparams['logs_images_folder'])
train_summary_writer = tf.summary.create_file_writer(hyperparams['logs_scalar_folder'])

"""
Prepare the tensorboard images
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
fsrcnn = FSRCNN(
    input_shape=(f_sub_lr, f_sub_lr, 1),
    d=config[config_to_run][0],
    s=config[config_to_run][1],
    m=config[config_to_run][2],
    upscaling=upscaling,
)

fsrcnn_optimizer = CustomAdam(learning_rate=1e-3)

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=fsrcnn_optimizer, net=fsrcnn)
ckpt_manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

fsrcnn.summary()
param_count = 0
for i in range(0, len(fsrcnn.layers), 2):
    param_count += fsrcnn.layers[i].count_params()
print("Number of parameters (PReLU not included):", param_count)

# Loading training and validation data
dat = np.load(data_path)
x = dat['x']
y = dat['y']

set5_patches = np.load("set5_7_21_3_3.npz")
set14_patches = np.load("set14_7_21_3_3.npz")
BSD200_patches = np.load("BSD200_7_21_3_3.npz")

set5 = pickle.load("set5.npz")
set14 = pickle.load("set14.npz")
BSD200 = pickle.load("BSD200.npz")


val_dat = np.load(val_data_path)
val_x = val_dat['x']
val_y = val_dat['y']

initial_log_step = 0
if args['initial_log_step'] is not None:
    initial_log_step = args['initial_log_step']

train(x, y, val_x, val_y, args['epochs'], ckpt_manager, initial_log_step=initial_log_step)