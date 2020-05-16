import tensorflow as tf
import pathlib
from tqdm import tqdm
import os
import numpy as np
import cv2
import pickle


def save_augmented_data(dataset, save_folder):
    data_dir = pathlib.Path(dataset)
    _, extension = os.path.splitext(os.listdir(data_dir)[3])
    for i in tqdm(data_dir.glob(f"*{extension}")):
        # Use grayscale because it is equivalent to first channel of yuv
        img = cv2.imread(str(i))  # cv2 uses bgr as default: https://stackoverflow.com/a/39316695
        ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = tf.expand_dims(ycrcb_image[:, :, 0], 2)
        for scale in [1, 0.9, 0.8, 0.7, 0.6]:
            for rot in range(4):
                hr = tf.keras.preprocessing.image.img_to_array(img, dtype=tf.float64)
                hr = tf.image.rot90(hr, k=rot)
                h, w, _ = hr.shape
                hr = tf.image.resize(
                    tf.identity(hr), (int(scale * h), int(scale * w)), method=tf.image.ResizeMethod.BICUBIC
                )
                if str(i).find("\\") == -1:
                    name = str(i).split("/")[2].split(".")[0]
                else:
                    name = str(i).split("\\")[2].split(".")[0]
                tf.keras.preprocessing.image.save_img(
                    f"{save_folder}/{name}_rot={rot*90}_scale={scale}{extension}", x=hr
                )


def generate_2(dataset_folder, output_path, f_sub_lr, aug, upscaling):
    import h5py
    import glob
    from PIL import Image

    h5_file = h5py.File(output_path, "w")
    lr_patches = []
    hr_patches = []

    for image_path in tqdm(sorted(glob.glob("{}/*".format(dataset_folder)))):
        hr = Image.open(image_path).convert("RGB")
        hr_images = []
        if aug:
            for s in [1.0, 0.9, 0.8, 0.7, 0.6]:
                for r in [0, 90, 180, 270]:
                    tmp = hr.resize((int(hr.width * s), int(hr.height * s)), resample=Image.BICUBIC)
                    tmp = tmp.rotate(r, expand=True)
                    hr_images.append(tmp)
        else:
            hr_images.append(hr)

        for hr in hr_images:
            hr_width = (hr.width // upscaling) * upscaling
            hr_height = (hr.height // upscaling) * upscaling
            hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
            lr = hr.resize((hr.width // upscaling, hr_height // upscaling), resample=Image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            for i in range(0, lr.shape[0] - f_sub_lr + 1, upscaling):
                for j in range(0, lr.shape[1] - f_sub_lr + 1, upscaling):
                    lr_patches.append(lr[i : i + f_sub_lr, j : j + f_sub_lr])
                    hr_patches.append(
                        hr[
                            i * upscaling : i * upscaling + f_sub_lr * upscaling,
                            j * upscaling : j * upscaling + f_sub_lr * upscaling,
                        ]
                    )

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset("lr", data=lr_patches)
    h5_file.create_dataset("hr", data=hr_patches)

    h5_file.close()


def convert_rgb_to_y(img, dim_order="hwc"):
    if dim_order == "hwc":
        return 16.0 + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.0
    else:
        return 16.0 + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.0


def convert_rgb_to_ycbcr(img, dim_order="hwc"):
    if dim_order == "hwc":
        y = 16.0 + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.0
        cb = 128.0 + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.0
        cr = 128.0 + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.0
    else:
        y = 16.0 + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.0
        cb = 128.0 + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.0
        cr = 128.0 + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.0
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def convert_ycbcr_to_rgb(img, dim_order="hwc"):
    if dim_order == "hwc":
        r = 298.082 * img[..., 0] / 256.0 + 408.583 * img[..., 2] / 256.0 - 222.921
        g = 298.082 * img[..., 0] / 256.0 - 100.291 * img[..., 1] / 256.0 - 208.120 * img[..., 2] / 256.0 + 135.576
        b = 298.082 * img[..., 0] / 256.0 + 516.412 * img[..., 1] / 256.0 - 276.836
    else:
        r = 298.082 * img[0] / 256.0 + 408.583 * img[2] / 256.0 - 222.921
        g = 298.082 * img[0] / 256.0 - 100.291 * img[1] / 256.0 - 208.120 * img[2] / 256.0 + 135.576
        b = 298.082 * img[0] / 256.0 + 516.412 * img[1] / 256.0 - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


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
            image[i * f_sub : (i + 1) * f_sub, j * f_sub : (j + 1) * f_sub] = p_reshape[:, :, 0]
            count += 1
    return image


def modcrop(image, scale=3):
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def create_pickle_from_folder(dataset, save_folder, upscaling):
    data_dir = pathlib.Path(dataset)
    _, extension = os.path.splitext(os.listdir(data_dir)[3])
    paths = np.array(list(data_dir.glob(f"*{extension}")))
    x = []
    y = []
    for path in tqdm(paths):
        img = cv2.imread(str(path))  # cv2 uses bgr as default: https://stackoverflow.com/a/39316695
        ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = tf.expand_dims(ycrcb_image[:, :, 0], 2)
        hr = tf.keras.preprocessing.image.img_to_array(img)
        hr = modcrop(hr, scale=3)
        h, w, _ = hr.shape
        new_w = int(w / upscaling)
        new_h = int(h / upscaling)
        lr = tf.image.resize(tf.identity(hr), (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC)
        x.append(tf.expand_dims(lr, 0) / tf.keras.backend.max(lr))
        y.append(tf.expand_dims(hr, 0) / tf.keras.backend.max(lr))

    pickle.dump({"x": x, "y": y}, open(save_folder, "wb"))


# save_augmented_data("dataset/Set14", "dataset/Set14-aug")
# create_pickle_from_folder("dataset/Set5", "../Set5.p", 3)
# create_pickle_from_folder("dataset/Set14", "../Set14.p", 3)
# create_pickle_from_folder("dataset/BSD200", "../BSD200.p", 3)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from matplotlib.transforms import Bbox
    from upscale import upscale_image
    from fsrcnn import FSRCNN
    from custom_adam import CustomAdam
    from PIL import Image


    def show_patch(img, x1, x2, y1, y2, zoom, loc, loc1, loc2):
        fig, ax = plt.subplots(figsize=[5, 4])
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        extent = [0, img.width, 0, img.height]
        ax.imshow(img, extent=extent)
        axins = zoomed_inset_axes(ax, zoom=zoom, loc=loc)  # zoom = 6
        axins.imshow(img, extent=extent, interpolation="nearest")
        # sub region of the original image
        axins.set_xlim(x1 * img.width, x2 * img.width)
        axins.set_ylim(y1 * img.height, y2 * img.height)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        # draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="0.5")
        plt.axis("off")
        ax.axis("off")
        #plt.draw()

    # prepare the demo image
    fsrcnn = FSRCNN(input_shape=(7, 7, 1), d=56, s=12, m=4, upscaling=3)
    fsrcnn_optimizer = CustomAdam(
        learning_rate=tf.constant(1e-3, dtype=tf.float32), learning_rate_deconv=tf.constant(1e-4, dtype=tf.float32)
    )

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=fsrcnn_optimizer, net=fsrcnn)
    ckpt_manager = tf.train.CheckpointManager(ckpt, "./tf_ckpts_3757_epochs", max_to_keep=3)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    dir = pathlib.Path("dataset/Set5")
    _, extension = os.path.splitext(os.listdir(dir)[0])
    files = list(dir.glob(f"*{extension}"))
    file = files[2]
    #for file in tqdm():
    name = "butterfly.png"
    img_pred, _, img_hr, img_bicubic, _, _ = upscale_image(fsrcnn, file, 3)
    show_patch(img_bicubic, x1=0.10, x2=0.30, y1=0.65, y2=0.85, zoom=2, loc=4, loc1=1, loc2=3)
    plt.savefig("results/bicubic_result_" + name, type="png", bbox_inches="tight", pad_inches=0, dpi=127.5)