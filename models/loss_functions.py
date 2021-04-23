import sys

import tensorflow as tf

sys.path.append("..")

from utils.evaluation_utils import dice_coef, tversky


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    score = tversky(y_true, y_pred)
    return tf.keras.backend.pow((1 - score), gamma)
