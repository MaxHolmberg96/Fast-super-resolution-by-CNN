from data import *
from fsrcnn import *

"""
Paths
"""
checkpoint_path = "training_checkpoints/cp.ckpt"
data_path = "data.npz"

"""
Hyperparameters for the model
"""
upscaling = 3
f_sub_lr = 7
f_sub_hr = f_sub_lr * upscaling
patch_stride = 4
nr_validation_samples = 70000
batch_size = 32
epochs = 3
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
indices = np.random.choice(np.arange(x.shape[0]), nr_validation_samples)
val_x = x[indices]
val_y = y[indices]
x = np.delete(x, indices, 0)
y = np.delete(y, indices, 0)
fsrcnn.load_weights(checkpoint_path)
fsrcnn.fit(x=x,
           y=y,
           epochs=epochs,
           batch_size=batch_size,
           validation_data=(val_x, val_y),
           callbacks=[cp_callback],
           shuffle=True)
#fsrcnn.evaluate(val_x, val_y)

img = tf.keras.preprocessing.image.load_img("dataset/General-100/im_8.bmp", color_mode="grayscale")
hr = tf.keras.preprocessing.image.img_to_array(img)
h, w, _ = hr.shape
new_w = int(w / upscaling)
new_h = int(h / upscaling)
lr = tf.image.resize(tf.identity(hr), (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC)

print(lr.shape)
tf.keras.preprocessing.image.save_img(path="lr.bmp", x=lr)
#patches, patches_shape = extract_patches(lr, f_sub_lr)

image = fsrcnn.predict(tf.expand_dims(lr, 0))
#image = put_togeheter_patches(patches_pred, patches_shape, f_sub_hr)
#print(image.shape)
tf.keras.preprocessing.image.save_img(path="upscaled_lr.bmp", x=image[0])