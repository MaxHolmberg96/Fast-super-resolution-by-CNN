import os
import pathlib

import numpy as np
from PIL import Image
from tqdm import tqdm

from data import convert_rgb_to_y, convert_rgb_to_ycbcr, convert_ycbcr_to_rgb
from fsrcnn import psnr


def upscale_large(fsrcnn, image_folder, output_folder, upscaling, rgb=True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    dir = pathlib.Path(image_folder)
    _, extension = os.path.splitext(os.listdir(dir)[0])
    for file in tqdm(dir.glob(f"*{extension}")):
        if str(file).find("\\") == -1:
            name = str(file).split("/")[-1]
        else:
            name = str(file).split("\\")[-1]
        img = upscale_image(fsrcnn, file, upscaling, rgb, downsample=False)
        img[0].save(os.path.join(output_folder, name))


def upscale(fsrcnn, image_folder, output_folder, upscaling, rgb=True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(os.path.join(output_folder, "original")):
        os.makedirs(os.path.join(output_folder, "original"))
    if not os.path.exists(os.path.join(output_folder, "low_res")):
        os.makedirs(os.path.join(output_folder, "low_res"))
    if not os.path.exists(os.path.join(output_folder, "upscaled")):
        os.makedirs(os.path.join(output_folder, "upscaled"))
    if not os.path.exists(os.path.join(output_folder, "bicubic")):
        os.makedirs(os.path.join(output_folder, "bicubic"))
    if not os.path.exists(os.path.join(output_folder, "togheter")):
        os.makedirs(os.path.join(output_folder, "togheter"))
    dir = pathlib.Path(image_folder)
    _, extension = os.path.splitext(os.listdir(dir)[0])
    psnr_pred = []
    psnr_bicubic = []
    for file in tqdm(dir.glob(f"*{extension}")):
        if str(file).find("\\") == -1:
            name = str(file).split("/")[-1]
        else:
            name = str(file).split("\\")[-1]
        img_pred, img_lr, img_hr, img_bicubic, img_all, psnr_images = upscale_image(fsrcnn, file, upscaling, rgb)
        psnr_pred.append(psnr_images[0])
        psnr_bicubic.append(psnr_images[1])
        img_pred.save(os.path.join(output_folder, "upscaled", name))
        img_lr.save(os.path.join(output_folder, "low_res", name))
        img_hr.save(os.path.join(output_folder, "original", name))
        img_bicubic.save(os.path.join(output_folder, "bicubic", name))
        img_all.save(os.path.join(output_folder, "togheter", name))
        with open(os.path.join(output_folder, "psnr.txt"), "w") as f:
            f.write("Predicted psnr: " + str(np.mean(psnr_pred)) + "\nBicubic psnr: " + str(np.mean(psnr_bicubic)))
        f.close()


def upscale_image(fsrcnn, file, upscaling, rgb=True, downsample=True):
    hr = Image.open(file).convert("RGB")
    hr_width = (hr.width // upscaling) * upscaling
    hr_height = (hr.height // upscaling) * upscaling
    hr_1 = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
    lr_1 = hr.resize((hr_width // upscaling, hr_height // upscaling), resample=Image.BICUBIC)
    hr = np.array(hr_1).astype(np.float32)
    lr = np.array(lr_1).astype(np.float32)
    lr = np.expand_dims(lr, 2)
    hr = np.expand_dims(hr, 0)
    lr = np.expand_dims(lr, 0)
    if downsample:
        bicubic = np.array(lr_1.resize((hr_width, hr_height), resample=Image.BICUBIC)).astype(np.float32)
        pred = fsrcnn.predict(convert_rgb_to_y(lr) / 255.0)
        pred = (pred * 255.0).squeeze(0).squeeze(-1)
        lr = np.squeeze(lr[0], 2)
    else:
        bicubic = np.array(hr_1.resize((hr_width * upscaling, hr_height * upscaling), resample=Image.BICUBIC)).astype(
            np.float32
        )
        pred = fsrcnn.predict(convert_rgb_to_y(np.expand_dims(hr, 3)) / 255.0)
        pred = (pred * 255.0).squeeze(0).squeeze(-1)

    if rgb:
        ycbcr = convert_rgb_to_ycbcr(bicubic)
        color_image = convert_ycbcr_to_rgb(np.array([pred, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0]))
        color_image = np.clip(color_image, 0.0, 255.0).astype(np.uint8)
        img_pred = Image.fromarray(color_image)
        if not downsample:
            return img_pred, bicubic

        lr = lr.astype(np.uint8)
        img_lr = Image.fromarray(lr)

        hr = (hr[0]).astype(np.uint8)
        img_hr = Image.fromarray(hr)

        bicubic = (bicubic).astype(np.uint8)
        img_bicubic = Image.fromarray(bicubic)

        all = np.hstack([hr, bicubic, color_image])
        img_all = Image.fromarray(all)
    else:
        img_pred = Image.fromarray(pred).convert("RGB")
        if not downsample:
            return img_pred, bicubic

        lr = convert_rgb_to_y(lr)
        img_lr = Image.fromarray(lr).convert("RGB")

        hr = convert_rgb_to_y(hr[0])
        img_hr = Image.fromarray(hr).convert("RGB")

        bicubic = convert_rgb_to_y(bicubic)
        img_bicubic = Image.fromarray(bicubic).convert("RGB")

        all = np.hstack([hr, bicubic, pred])
        img_all = Image.fromarray(all).convert("RGB")
    psnr_pred = psnr(y_pred=np.array(img_pred) / 255.0, y_true=np.array(img_hr) / 255.0, clip=False)
    psnr_bicubic = psnr(y_pred=np.array(img_bicubic) / 255.0, y_true=np.array(img_hr) / 255.0, clip=False)

    return img_pred, img_lr, img_hr, img_bicubic, img_all, (psnr_pred, psnr_bicubic)


def psnr_given_two_paths(path1, path2, upscaling):
    img1 = Image.open(path1).convert("RGB")
    img2 = Image.open(path2).convert("RGB")

    hr_width = (img1.width // upscaling) * upscaling
    hr_height = (img1.height // upscaling) * upscaling
    img1 = img1.resize((hr_width, hr_height), resample=Image.BICUBIC)

    hr_width = (img2.width // upscaling) * upscaling
    hr_height = (img2.height // upscaling) * upscaling
    img2 = img2.resize((hr_width, hr_height), resample=Image.BICUBIC)

    img1 = np.array(img1).astype(np.float32)
    img1 = convert_rgb_to_y(img1) / 255.0

    img2 = np.array(img2).astype(np.float32)
    img2 = convert_rgb_to_y(img2) / 255.0

    return psnr(y_pred=np.array(np.expand_dims(img1, 2)), y_true=np.array(np.expand_dims(img2, 2)), clip=False)

def psnr_given_two_folders(folder1, folder2, upscaling):
    dir = pathlib.Path(folder1)
    _, extension = os.path.splitext(os.listdir(dir)[0])
    list1 = list(dir.glob(f"*{extension}"))
    dir = pathlib.Path(folder2)
    _, extension = os.path.splitext(os.listdir(dir)[0])
    list2 = list(dir.glob(f"*{extension}"))
    p = []
    for file1, file2 in tqdm(zip(list1, list2)):
        p.append(psnr_given_two_paths(file1, file2, upscaling))
    return np.mean(p)



def resize_modcrop(path, output, upscaling):
    hr = Image.open(path).convert("RGB")
    hr_width = (hr.width // upscaling) * upscaling
    hr_height = (hr.height // upscaling) * upscaling
    hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
    hr.save(output)