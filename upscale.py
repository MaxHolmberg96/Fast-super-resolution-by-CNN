import pathlib
from PIL import Image
import os
from data import *
from tqdm import tqdm

def upscale(fsrcnn, image_folder, output_folder, upscaling):
    from run import PSNR
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
    _, extension = os.path.splitext(os.listdir(dir)[3])
    psnr = []
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
        psnr.append(PSNR(fsrcnn, lr, hr))

        pred = np.squeeze(pred[0], 2) * 255.0
        img = Image.fromarray(pred).convert("RGB")
        img.save(os.path.join(output_folder, "upscaled", str(file).split("\\")[-1]))

        lr = np.squeeze(lr[0], 2) * 255.0
        img = Image.fromarray(lr).convert("RGB")
        img.save(os.path.join(output_folder, "low_res", str(file).split("\\")[-1]))

        hr = np.squeeze(hr[0], 2) * 255.0
        img = Image.fromarray(hr).convert("RGB")
        img.save(os.path.join(output_folder, "original", str(file).split("\\")[-1]))

        bicubic = np.squeeze(bicubic, 2) * 255.0
        img = Image.fromarray(bicubic).convert("RGB")
        img.save(os.path.join(output_folder, "bicubic", str(file).split("\\")[-1]))

        all = np.hstack([hr, bicubic, pred])
        img = Image.fromarray(all).convert("RGB")
        img.save(os.path.join(output_folder, "togheter", str(file).split("\\")[-1]))



    with open(os.path.join(output_folder, "psnr.txt"), "w") as f:
        f.write(str(np.mean(psnr)))
    f.close()