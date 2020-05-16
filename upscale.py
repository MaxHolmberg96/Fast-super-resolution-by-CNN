import pathlib
from PIL import Image
import os
from data import *
from tqdm import tqdm
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
        img.save(os.path.join(output_folder, name))


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
    psnrs = []
    for file in tqdm(dir.glob(f"*{extension}")):
        if str(file).find("\\") == -1:
            name = str(file).split("/")[-1]
        else:
            name = str(file).split("\\")[-1]
        img_pred, img_lr, img_hr, img_bicubic, img_all, psnr_image = upscale_image(fsrcnn, file, upscaling, rgb)
        psnrs.append(psnr_image)
        img_pred.save(os.path.join(output_folder, "upscaled", name))
        img_lr.save(os.path.join(output_folder, "low_res", name))
        img_hr.save(os.path.join(output_folder, "original", name))
        img_bicubic.save(os.path.join(output_folder, "bicubic", name))
        img_all.save(os.path.join(output_folder, "togheter", name))
        with open(os.path.join(output_folder, "psnr.txt"), "w") as f:
            f.write(str(np.mean(psnrs)))
        f.close()


def upscale_image(fsrcnn, file, upscaling, rgb=True, downsample=True):
    hr = Image.open(file).convert('RGB')
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
        pred = fsrcnn.predict(convert_rgb_to_y(lr) / 255.)
        pred = (pred * 255.).squeeze(0).squeeze(-1)
        lr = np.squeeze(lr[0], 2)
    else:
        bicubic = np.array(hr_1.resize((hr_width * upscaling, hr_height * upscaling), resample=Image.BICUBIC)).astype(np.float32)
        pred = fsrcnn.predict(convert_rgb_to_y(np.expand_dims(hr, 3)) / 255.)
        pred = (pred * 255.).squeeze(0).squeeze(-1)

    if rgb:
        ycbcr = convert_rgb_to_ycbcr(bicubic)
        color_image = convert_ycbcr_to_rgb(np.array([pred, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0]))
        color_image = np.clip(color_image, 0.0, 255.0).astype(np.uint8)
        img_pred = Image.fromarray(color_image)
        if not downsample:
            return img_pred

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
            return img_pred

        lr = convert_rgb_to_y(lr)
        img_lr = Image.fromarray(lr).convert("RGB")

        hr = convert_rgb_to_y(hr[0])
        img_hr = Image.fromarray(hr).convert("RGB")

        bicubic = convert_rgb_to_y(bicubic)
        img_bicubic = Image.fromarray(bicubic).convert("RGB")

        all = np.hstack([hr, bicubic, pred])
        img_all = Image.fromarray(all).convert("RGB")
    psnr_image = psnr(y_pred=np.array(img_pred) / 255., y_true=np.array(img_hr) / 255., clip=False)

    return img_pred, img_lr, img_hr, img_bicubic, img_all, psnr_image
