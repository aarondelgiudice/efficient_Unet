import collections

import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------
# evaluation metrics
def IoU_score(y_true, y_pred):
    """ IOU score == true positives / (true positives + false positives + false negatives) """
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    
    true_positives = tf.keras.backend.sum(y_true_pos * y_pred_pos)
    false_positives = tf.keras.backend.sum((1 - y_true_pos) * y_pred_pos)
    false_negatives = tf.keras.backend.sum(y_true_pos * (1 - y_pred_pos))
    
    score = tf.math.divide_no_nan(
        true_positives, (true_positives + false_positives + false_negatives))
    
    return score

def Recall(y_true, y_pred):
    """ Recall == true positives / (true positives + false negatives) """
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    
    true_positives = tf.keras.backend.sum(y_true_pos * y_pred_pos)
    false_negatives = tf.keras.backend.sum(y_true_pos * (1 - y_pred_pos))
    
    score = tf.math.divide_no_nan(
        true_positives, (true_positives + false_negatives))
    
    return score


def Precision(y_true, y_pred):
    """ Precision == true positives / (true positives + false positives) """
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    
    true_positives = tf.keras.backend.sum(y_true_pos * y_pred_pos)
    false_positives = tf.keras.backend.sum((1 - y_true_pos) * y_pred_pos)
    
    score = tf.math.divide_no_nan(
        true_positives, (true_positives + false_positives))
    
    return score


def F1_score(y_true, y_pred):
    """ F1-Score == 2 * (precision * recall) / (precision + recall) """
    recall = Recall(y_true, y_pred)
    precision = Precision(y_true, y_pred)
    
    score = 2 * tf.math.divide_no_nan(
        (precision * recall), (precision + recall))
    
    return score


def dice_coef(y_true, y_pred, smooth=1):
    """
    dice coefficient == 2 * true positives / 2 * (true positives + false positives + false negatives)
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    
    dice_coefficient = tf.math.divide_no_nan(
        (2. * intersection + smooth),
        (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))
    
    return dice_coefficient


def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    """
    tversky index = 
        (true positives + smooth) /
        (true positives + (alpha * false negatives) + ((1 - alpha) * false positives) + smooth)
    """
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    true_pos = tf.keras.backend.sum(y_true_pos * y_pred_pos)
    false_neg = tf.keras.backend.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.keras.backend.sum((1 - y_true_pos) * y_pred_pos)
    
    tversky_index = tf.math.divide_no_nan(
        (true_pos + smooth),
        (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth))
    
    return tversky_index


# model utils
def load_model(weights):
    return tf.keras.models.load_model(
        weights,
        compile=False,
        custom_objects={'leaky_relu': tf.nn.leaky_relu})


# -----------------------------------------------------------------------------
# training metrics
# -----------------------------------------------------------------------------
metric = collections.namedtuple("metric", ["name", "metric"])

metrics = [
    metric("Loss", None),
    metric("Mean Absolute Error", tf.keras.metrics.MeanAbsoluteError()),
    metric("IoU Score", IoU_score),
    metric("F1-Score", F1_score),
    metric("Precision", Precision),
    metric("Recall", Recall),
    metric("Dice Coefficient", dice_coef),
    metric("Tversky Index", tversky),
]
