import pathlib
from PIL import Image
import os
from data import *
from tqdm import tqdm

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
    dir = pathlib.Path(image_folder)
    _, extension = os.path.splitext(os.listdir(dir)[3])
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
        lr = np.expand_dims(lr, 0)
        bicubic = convert_rgb_to_y(np.array(lr_1.resize((hr_width, hr_height), resample=Image.BICUBIC)).astype(np.float32))
        bicubic /= 255.
        bicubic = np.expand_dims(bicubic, 2)


        pred = fsrcnn.predict(lr)
        img = Image.fromarray(np.squeeze(pred[0], 2) * 255.0).convert("RGB")
        img.save(os.path.join(output_folder, "upscaled", str(file).split("\\")[-1]))

        img = Image.fromarray(np.squeeze(lr[0], 2) * 255.0).convert("RGB")
        img.save(os.path.join(output_folder, "low_res", str(file).split("\\")[-1]))

        img = Image.fromarray(np.squeeze(hr, 2) * 255.0).convert("RGB")
        img.save(os.path.join(output_folder, "original", str(file).split("\\")[-1]))

        img = Image.fromarray(np.squeeze(bicubic, 2) * 255.0).convert("RGB")
        img.save(os.path.join(output_folder, "bicubic", str(file).split("\\")[-1]))