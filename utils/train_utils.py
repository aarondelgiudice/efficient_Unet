import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append("..")
from utils.utils import visualize


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------
def decode_predictions(y_pred, labels):
    max_filter = np.argmax(y_pred, axis=-1)
    
    output = np.zeros((*max_filter.shape, 3))
    
    for i in labels:
        output[max_filter == i.categoryId] = i.color
    
    return output


# image preprocessing ---------------------------------------------------------
def normalize(a):
    # normalizing the images to [0, 1]
    a = tf.math.divide_no_nan(
        (a - tf.reduce_min(a)),
        (tf.reduce_max(a) - tf.reduce_min(a)))
    return a


def normalize_x(x, method="imagenet"):
    # normalize to expected imagenet values
    x = tf.cast(x, tf.float32)
    
    if method is "imagenet": 
        x = tf.keras.applications.imagenet_utils.preprocess_input(x)
    
    else:
        # set range to [0, 1]
        x = normalize(x)

    return x


def encode_y(y, labels):
    """one hot encode pixel values to category IDs"""
    y_adj = tf.zeros(shape=y.shape[:2])
    y_adj = tf.cast(y_adj, tf.int32)

    # convert RGB pixel values to category IDs as ints
    for i in labels:
        y_adj = tf.where(
            tf.reduce_all(y == i.color, axis=-1),
            i.categoryId, y_adj
        )
    
    # breakout category IDs into one-hot encoded arrays
    IDs = set([i.categoryId for i in labels])
    y_adj = [tf.where(y_adj == i, 1, 0) for i in range(len(IDs))]

    y_adj = tf.stack(y_adj, axis=-1)
    y_adj = tf.cast(y_adj, tf.float32)

    return y_adj


# train augmentations ---------------------------------------------------------
def image_translate(image, mask, magnitude=0.25):
    """ translate images by 20% """
    
    x_max, y_max = image.shape[0] * magnitude, image.shape[1] * magnitude
    
    # translate [-max val: max val]
    x_translation = tf.random.uniform((), minval=(x_max * -1), maxval=x_max)
    y_translation = tf.random.uniform((), minval=(y_max * -1), maxval=y_max)
    
    # apply x, y translation and fill with 0
    translated_image = tfa.image.translate(
        image,
        translations=[x_translation, y_translation],
        interpolation='nearest',
        fill_mode='constant',
        fill_value=-0.0
    )
    
    translated_mask = tfa.image.translate(
        mask,
        translations=[x_translation, y_translation],
        interpolation='nearest',
        fill_mode='constant',
        fill_value=0.0,
    )
    
    return translated_image, translated_mask


def resize_and_random_crop(image, mask, shape=(256, 256, 3), factor=1.2):
    # resize image
    scaled_h, scaled_w = int(shape[0] * scalar), int(shape[1] * factor)

    image = tf.image.resize(
        image, [scaled_h, scaled_w],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    mask = tf.image.resize(
        mask, [scaled_h, scaled_w],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    # randomly cropping to original image size
    stacked_image = tf.stack([image, mask], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, *shape])

    return cropped_image[0], cropped_image[1]


@tf.function()
def random_jitter(
    image, mask,
    shape=(512, 512, 3),
    brightness_max_delta=0.2,
    contrast_min=0.2,
    contrast_max=0.5,
    gaussian_filter=(16, 16),
    gaussian_sigma=50,
    max_image_scale=0.2,
    translation=0.2,
):
    if tf.random.uniform(()) > 0.5:
        # random mirroring of 50% of images
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        
    if max_image_scale is not None:
        # randomly cropping from +20% to original image size
        image_scale = 1 + tf.random.uniform(shape=(), minval=0, maxval=max_image_scale)
        image, mask = resize_and_random_crop(image=image, mask=mask, shape=shape, factor=image_scale)
    
    if tf.random.uniform(()) > 0.5:
        # randomly reverse RGB channels
        image = tf.reverse(image, axis=[-1])
    
    if (brightness_max_delta and contrast_min and contrast_max) is not None:
        if tf.random.uniform(()) > 0.5:
            # random brightness, contrast of 50% of images
            image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
            image = tf.image.random_contrast(image, lower=contrast_min, upper=contrast_max)
    
    if (gaussian_filter and gaussian_sigma) is not None:
        if tf.random.uniform(()) > 0.75:
            # random bluring of 25% of images
            image = tfa.image.gaussian_filter2d(
                image, filter_shape=gaussian_filter, sigma=gaussian_sigma)
    
    if translation is not None:
        if tf.random.uniform(()) > 0.5:
            # randomly translate image by [-20% : 20%]
            image, mask = image_translate(image, mask, magnitude=translation)

    return image, mask


# loading images into data generators -----------------------------------------
def load_image(filepath, shape=(512, 512, 3), dtype=None):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=shape[-1])
    
    if shape is not None:
        image = tf.image.resize(
            image, shape[:2],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
    if dtype is not None:
        image = tf.cast(image, dtype)

    return image


def parse_file_path(
    fp,
    from_tensor=False,
    image_dir="images",
    mask_dir="masks",
    image_set="leftImg8bit",
    mask_set="gtFine_color",
):
    if from_tensor:
        fp = tf.strings.regex_replace(
            input=fp, pattern=image_dir, rewrite=mask_dir)
        fp = tf.strings.regex_replace(
            input=fp, pattern=image_set, rewrite=mask_set)

    else:
        fp = fp.replace(image_dir, mask_dir)
        fp = fp.replace(image_set, mask_set)

    return fp


def load(img_path: str, shape=(256, 256, 3)):
    image = load_image(img_path, shape=shape, dtype=tf.float32)
    
    mask_path = parse_file_path(fp=img_path, from_tensor=True)
    
    mask = load_image(mask_path, shape=shape, dtype=tf.float32)

    return image, mask


# training callbacks ----------------------------------------------------------
def generate_images(model, x_data, labels, y_data=None, fp=None, metrics_labels=None):
    # predict images
    y_pred = model(tf.expand_dims(x_data, axis=0), training=False)

    # convert ndim -> 3dim
    pred_mask = decode_predictions(y_pred, labels)

    # human readable
    x_image = normalize(x_data)
    pred_mask = tf.squeeze(normalize(pred_mask))

    if y_data is not None:
        # evaluate images
        metrics_values = model.evaluate(
            tf.expand_dims(x_data, axis=0),
            tf.expand_dims(y_data, axis=0),
            verbose=False)

        metrics = {k: v for k, v in zip(metrics_labels, metrics_values)}

        # convert ndim -> 3dim
        #true_mask = decode_predictions(tf.expand_dims(y_data, axis=0), colormap)
        true_mask = decode_predictions(y_data, labels)
        true_mask = normalize(true_mask)  # human readable

        visualize(
            title=f"{[f'{k} : {np.round(v, 4)}' for k, v in metrics.items()]}",
            input_image=x_image, ground_truth=true_mask, predicted_image=pred_mask, savefig=fp)
    
    else:
        visualize(input_image=x_image, predicted_image=pred_mask, savefig=fp)

    return y_pred


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, train_sample, labels, metrics_labels=None, test_sample=None, output=None):
        self.model = model
        self.train_x, self.train_y = train_sample
        self.test_x, self.test_y = test_sample if test_sample is not None else (None, None)
        self.output = output if output is not None else os.getcwd()
        self.labels = labels
        self.metrics_labels = metrics_labels
        return

    def on_epoch_end(self, epoch, logs=None):
        IPython.display.clear_output(wait=True)  # ! delete if in .py script
        
        epoch += 1  # human readable

        # confirm necessary paths exist
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        # plot and save training sample
        generate_images(
            model=self.model, x_data=self.train_x, y_data=self.train_y,
            labels=self.labels, metrics_labels=self.metrics_labels,
            fp=os.path.join(self.output, f"train_epoch-{epoch}.png"))

        if self.test_x is not None and self.test_y is not None:
            generate_images(
                model=self.model, x_data=self.test_x, y_data=self.test_y,
                labels=self.labels, metrics_labels=self.metrics_labels,
                fp=os.path.join(self.output, f"test_epoch-{epoch}.png"))

        return
