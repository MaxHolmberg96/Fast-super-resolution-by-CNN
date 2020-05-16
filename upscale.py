import pathlib
from PIL import Image
import os
from data import *
from tqdm import tqdm
from fsrcnn import psnr

def upscale(fsrcnn, image_folder, output_folder, upscaling):
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
        hr = Image.open(file).convert('RGB')
        hr_width = (hr.width // upscaling) * upscaling
        hr_height = (hr.height // upscaling) * upscaling
        hr_1 = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
        lr_1 = hr.resize((hr_width // upscaling, hr_height // upscaling), resample=Image.BICUBIC)
        hr = np.array(hr_1).astype(np.float32)
        lr = np.array(lr_1).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)
        hr /= 255.
        lr /= 255.
        hr = np.expand_dims(hr, 2)
        lr = np.expand_dims(lr, 2)
        hr = np.expand_dims(hr, 0)
        lr = np.expand_dims(lr, 0)
        bicubic = convert_rgb_to_y(np.array(lr_1.resize((hr_width, hr_height), resample=Image.BICUBIC)).astype(np.float32))
        bicubic /= 255.
        bicubic = np.expand_dims(bicubic, 2)


        pred = fsrcnn.predict(lr)
        psnrs.append(psnr(y_pred=pred, y_true=hr, clip=False))

        pred = np.squeeze(pred[0], 2) * 255.0
        img = Image.fromarray(pred).convert("RGB")
        if str(file).find("\\") == -1:
            name = str(file).split("/")[-1]
        else:
            name = str(file).split("\\")[-1]
        img.save(os.path.join(output_folder, "upscaled", name))

        lr = np.squeeze(lr[0], 2) * 255.0
        img = Image.fromarray(lr).convert("RGB")
        img.save(os.path.join(output_folder, "low_res", name))

        hr = np.squeeze(hr[0], 2) * 255.0
        img = Image.fromarray(hr).convert("RGB")
        img.save(os.path.join(output_folder, "original", name))

        bicubic = np.squeeze(bicubic, 2) * 255.0
        img = Image.fromarray(bicubic).convert("RGB")
        img.save(os.path.join(output_folder, "bicubic", name))

        all = np.hstack([hr, bicubic, pred])
        img = Image.fromarray(all).convert("RGB")
        img.save(os.path.join(output_folder, "togheter", name))



    with open(os.path.join(output_folder, "psnr.txt"), "w") as f:
        f.write(str(np.mean(psnrs)))
    f.close()

def upscale_large(fsrcnn, image_folder, output_folder, upscaling):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    dir = pathlib.Path(image_folder)
    _, extension = os.path.splitext(os.listdir(dir)[0])
    for file in tqdm(dir.glob(f"*{extension}")):
        hr = Image.open(file).convert('RGB')
        hr = np.array(hr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        hr /= 255.
        hr = np.expand_dims(hr, 2)
        hr = np.expand_dims(hr, 0)
        pred = fsrcnn.predict(hr)
        pred = np.squeeze(pred[0], 2) * 255.0
        img = Image.fromarray(pred).convert("RGB")
        if str(file).find("\\") == -1:
            name = str(file).split("/")[-1]
        else:
            name = str(file).split("\\")[-1]
        img.save(os.path.join(output_folder, name))

def upscale_rgb(fsrcnn, image_folder, output_folder, upscaling):
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
    for file in tqdm(dir.glob(f"*{extension}")):
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
        bicubic = np.array(lr_1.resize((hr_width, hr_height), resample=Image.BICUBIC)).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(bicubic)


        if str(file).find("\\") == -1:
            name = str(file).split("/")[-1]
        else:
            name = str(file).split("\\")[-1]

        pred = fsrcnn.predict(convert_rgb_to_y(lr) / 255.)
        pred = (pred * 255.).squeeze(0).squeeze(-1)
        color_image = convert_ycbcr_to_rgb(np.array([pred, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0]))
        color_image = np.clip(color_image, 0.0, 255.0)

        color_image = (color_image).astype(np.uint8)
        img = Image.fromarray(color_image)
        img.save(os.path.join(output_folder, "upscaled", name))

        lr = (np.squeeze(lr[0], 2)).astype(np.uint8)
        img = Image.fromarray(lr)
        img.save(os.path.join(output_folder, "low_res", name))

        hr = (hr[0]).astype(np.uint8)
        img = Image.fromarray(hr)
        img.save(os.path.join(output_folder, "original", name))

        bicubic = (bicubic).astype(np.uint8)
        img = Image.fromarray(bicubic)
        img.save(os.path.join(output_folder, "bicubic", name))

        all = np.hstack([hr, bicubic, color_image])
        img = Image.fromarray(all)
        img.save(os.path.join(output_folder, "togheter", name))
