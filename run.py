import argparse
import datetime
import os
import pathlib

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image

import data
from custom_adam import CustomAdam
from fsrcnn import FSRCNN

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
hyperparams = {
    "adam_alpha": 1e-3,
    "update_losses": 1000,
    "logs_images_folder": "tensorboard_logs/" + current_time + "_images/",
    "logs_scalar_folder": "tensorboard_logs/" + current_time + "/train",
}


@tf.function()
def train_step(x, y):
    with tf.GradientTape() as g:
        mse = fsrcnn_loss(fsrcnn, x, y)
    gradients = g.gradient(mse, fsrcnn.trainable_variables)
    fsrcnn_optimizer.apply_gradients(zip(gradients, fsrcnn.trainable_variables))
    return gradients


mse = tf.keras.losses.MeanSquaredError()


# @tf.function()
def fsrcnn_loss(model, x, y_true):
    y_pred = model(x)
    return mse(y_true, y_pred)


@tf.function(experimental_relax_shapes=True)
def PSNR(model, x, y_true):
    y_pred = model(x)
    ps = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return tf.clip_by_value(ps, clip_value_min=0, clip_value_max=99.9)


def train(x, y, val_x, val_y, epochs, ckpt_manager, shuffle=True):
    import time
    from tqdm import tqdm

    if shuffle:
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

    if args["continue"]:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        if ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
    else:
        print("Initializing from scratch.")

    update_step = 0
    for epoch in range(epochs):
        start = time.time()
        offset = 0
        iterator = tqdm(range(x.shape[0] // hyperparams["batch_size"]))
        for i in iterator:
            batch_x = x[offset : offset + hyperparams["batch_size"]]
            batch_y = y[offset : offset + hyperparams["batch_size"]]
            grads = train_step(batch_x, batch_y)
            if update_step % hyperparams["update_losses"] == 0:
                loss = fsrcnn_loss(fsrcnn, batch_x, batch_y)
                val_loss = fsrcnn_loss(fsrcnn, val_x, val_y)
                psnr = np.mean(PSNR(fsrcnn, batch_x, batch_y))
                val_psnr = np.mean(PSNR(fsrcnn, val_x, val_y))

                iterator.set_description(
                    "\nloss: {:.5f}, val_loss: {:.5f}, psnr: {:.5f}, val_psnr: {:.5f}".format(
                        loss, val_loss, psnr, val_psnr
                    )
                )
                # Write to tensorboard
                write_batch_summaries(loss, val_loss, psnr, val_psnr, update_step)

            offset += hyperparams["batch_size"]
            update_step += 1

        ckpt.step.assign_add(1)
        save_path = ckpt_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        write_epoch_summaries(grads, fsrcnn, epoch)

        if args['include_test'] and epoch % 50 == 0:
            test_psnr_patches = np.mean(PSNR(fsrcnn, set5_patches["x"], set5_patches["y"]))
            test_psnr_patches += np.mean(PSNR(fsrcnn, set14_patches["x"], set14_patches["y"]))
            test_psnr_patches += np.mean(PSNR(fsrcnn, BSD200_patches["x"], BSD200_patches["y"]))
            test_psnr_patches = test_psnr_patches / 3
            with train_summary_writer.as_default():
                tf.summary.scalar("Average test PSNR", test_psnr_patches, epoch)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))


def write_epoch_summaries(grads, model, epoch):
    for i, show_image in enumerate(show_images):
        pred = model.predict(show_image)
        pred = (pred * 255.0).squeeze(0).squeeze(-1)
        color_image = data.convert_ycbcr_to_rgb(
            np.array([pred, ycbcr_images[i][..., 1], ycbcr_images[i][..., 2]]).transpose([1, 2, 0])
        )
        color_image = np.expand_dims(np.clip(color_image, 0.0, 255.0) / 255.0, 0)
        with image_summary_writer.as_default():
            tf.summary.image(str(i), tf.concat(hr_and_bicubic[i] + [color_image], 0), step=epoch)

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
        tf.summary.scalar("mse", loss, step=update_log_step)
        tf.summary.scalar("val_mse", val_loss, step=update_log_step)
        tf.summary.scalar("psnr", psnr, step=update_log_step)
        tf.summary.scalar("val_psnr", val_psnr, step=update_log_step)


ap = argparse.ArgumentParser()
ap.add_argument("-train_path", "--train_path", required=True, help="Path to the training data")
ap.add_argument("-val_path", "--val_path", required=True, help="Path to the validation data")
ap.add_argument("-batch_size", "--batch_size", required=False, type=int, default=64, help="Batch size during training")
ap.add_argument("-epochs", "--epochs", required=False, type=int, default=100, help="Number of epochs")
ap.add_argument("-f_sub_lr", "--f_sub_lr", required=False, type=int, default=7, help="Size of lr patches")
ap.add_argument("-upscaling", "--upscaling", required=False, type=int, default=3, help="Upscaling factor")
ap.add_argument("-fast", "--fast", required=False, action="store_true", help="Set this if you want to train FSRCNN-s")
ap.add_argument("-weights", "--weights", required=False, help="path to the weights to use")
ap.add_argument("-continue", "--continue", required=False, action="store_true", help="If we want to continue training")
ap.add_argument("-include_test", "--include_test", required=False, action="store_true",
                help="If we want to output test images (Set5) in tensorboard while training")
args = vars(ap.parse_args())

"""
Parameters
"""
data_path = args["train_path"]
val_data_path = args["val_path"]
hyperparams["batch_size"] = args["batch_size"]
upscaling = args["upscaling"]
f_sub_lr = args["f_sub_lr"]
f_sub_hr = f_sub_lr * args["upscaling"]
batch_size = args["batch_size"]
epochs = args["epochs"]
config = [(56, 12, 4), (32, 5, 1)]
config_to_run = 1 if args['fast'] else 0
image_summary_writer = tf.summary.create_file_writer(hyperparams["logs_images_folder"])
train_summary_writer = tf.summary.create_file_writer(hyperparams["logs_scalar_folder"])

"""
Prepare the tensorboard images
"""
data_dir = pathlib.Path("dataset/Set5/")
hr_and_bicubic = []
show_images = []
ycbcr_images = []
_, extension = os.path.splitext(os.listdir(data_dir)[3])
for file in tqdm(data_dir.glob(f"*{extension}")):
    hr = Image.open(file).convert("RGB")
    hr_width = (hr.width // upscaling) * upscaling
    hr_height = (hr.height // upscaling) * upscaling
    hr_1 = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
    lr_1 = hr.resize((hr_width // upscaling, hr_height // upscaling), resample=Image.BICUBIC)
    hr = np.array(hr_1).astype(np.float32)
    lr = np.array(lr_1).astype(np.float32)
    lr = data.convert_rgb_to_y(lr)
    hr /= 255.0
    lr /= 255.0
    # hr = np.expand_dims(hr, 2)
    lr = np.expand_dims(lr, 2)
    show_images.append(np.expand_dims(lr, 0))
    bicubic = np.array(lr_1.resize((hr_width, hr_height), resample=Image.BICUBIC)).astype(np.float32)
    ycbcr = data.convert_rgb_to_ycbcr(bicubic)
    ycbcr_images.append(ycbcr)
    bicubic /= 255.0
    hr_and_bicubic.append([np.expand_dims(hr, 0), np.expand_dims(bicubic, 0)])


# Making the model
fsrcnn = FSRCNN(
    d=config[config_to_run][0],
    s=config[config_to_run][1],
    m=config[config_to_run][2],
    upscaling=upscaling,
)

fsrcnn_optimizer = CustomAdam(
    learning_rate=tf.constant(hyperparams["adam_alpha"], dtype=tf.float32),
    learning_rate_deconv=tf.constant(hyperparams["adam_alpha"] / 10, dtype=tf.float32),
)

if args['weights'] is None:
    weights = "weights_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
else:
    weights = args["weights"]
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=fsrcnn_optimizer, net=fsrcnn)
ckpt_manager = tf.train.CheckpointManager(ckpt, weights, max_to_keep=3)

fsrcnn.summary()
param_count = 0
for i in range(0, len(fsrcnn.layers), 2):
    param_count += fsrcnn.layers[i].count_params()
print("Number of parameters (PReLU not included):", param_count)

# Loading training and validation data
with h5py.File(data_path, "r") as f:
    x = np.array(f["lr"])
    y = np.array(f["hr"])
    x = np.expand_dims(x, 3) / 255.0
    y = np.expand_dims(y, 3) / 255.0

with h5py.File(val_data_path, "r") as f:
    val_x = np.array(f["lr"])
    val_y = np.array(f["hr"])
    val_x = np.expand_dims(val_x, 3) / 255.0
    val_y = np.expand_dims(val_y, 3) / 255.0

if args['include_test']:
    with h5py.File("Set5_7_21_3_3.h5") as f:
        set5_patches = {}
        set5_patches["x"] = np.array(f["lr"])
        set5_patches["y"] = np.array(f["hr"])
        set5_patches["x"] = np.expand_dims(set5_patches["x"], 3) / 255.0
        set5_patches["y"] = np.expand_dims(set5_patches["y"], 3) / 255.0

    with h5py.File("Set14_7_21_3_3.h5") as f:
        set14_patches = {}
        set14_patches["x"] = np.array(f["lr"])
        set14_patches["y"] = np.array(f["hr"])
        set14_patches["x"] = np.expand_dims(set14_patches["x"], 3) / 255.0
        set14_patches["y"] = np.expand_dims(set14_patches["y"], 3) / 255.0

    with h5py.File("BSD200_7_21_3_3.h5") as f:
        BSD200_patches = {}
        BSD200_patches["x"] = np.array(f["lr"])
        BSD200_patches["y"] = np.array(f["hr"])
        BSD200_patches["x"] = np.expand_dims(BSD200_patches["x"], 3) / 255.0
        BSD200_patches["y"] = np.expand_dims(BSD200_patches["y"], 3) / 255.0

train(x, y, val_x, val_y, args["epochs"], ckpt_manager)
