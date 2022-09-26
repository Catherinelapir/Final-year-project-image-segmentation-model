import numpy as np
import keras.backend_config as K
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import normalize
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.models import load_model


def IoU(y_true, y_pred):
    smooth = 1e-5
    y_true_f = float(K.flatten(y_true))
    y_pred_f = float(K.flatten(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth
    )


def IoU_Loss(y_true, y_pred):
    return 1 - IoU(y_true, y_pred)


def load_model(path):
    model = tf.keras.models.load_model(
        path, custom_objects={"IoU": IoU_Loss}, compile=False
    )
    model.compile(optimizer="adam", loss=IoU_Loss, metrics=[IoU, "binary_accuracy"])
    return model


def predict(img_path, model):
    img = Image.open(img_path)
    print(f"Image type: {type(img)}")
    img = img.resize((256, 256))
    img = img_to_array(img)
    img = normalize(img, axis=0)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction[0, :, :, 0]
