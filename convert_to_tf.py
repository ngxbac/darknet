import numpy as np
import os
from yolov4 import YOLOv4, decode
from common import load_weights
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--yolo_weights', default='ufpr_char_sgdr_mixup_mosac/yolov4-ufpr-char_last.weights')
parser.add_argument('--tf_models', default='ufpr_char_sgdr_mixup_mosac')
parser.add_argument('--input_width', default=416, type=int)
parser.add_argument('--input_height', default=416, type=int)

args = parser.parse_args()


if __name__ == '__main__':
    STRIDES = [8, 16, 32]
    # anchor for char recognition
    # ANCHORS = [6, 65,   9, 42,  13, 55,  12, 64,  14, 62,  22, 41,  15, 63,  14, 70,  16, 73]

    # anchor for CCPD plate detection
    # ANCHORS = [85, 19, 117, 24, 131, 30, 130, 39, 167, 33, 133, 55, 186, 42, 175, 64, 246, 51]

    # anchor for UFPR plate detection
    # ANCHORS = [65, 24,  83, 30,  87, 43, 108, 38, 105, 42, 110, 46, 145, 37, 128, 45, 139, 55]

    # anchor for VNSynthesis plate detection
    ANCHORS = [12, 46,  16, 41,  17, 47,  20, 46,  23, 45,  22, 53,  12,104,  27, 50,  30, 57]
    ANCHORS = np.asarray(ANCHORS).reshape((3, 3, 2))
    NUM_CLASS = 36
    XYSCALE = [1.2, 1.1, 1.05]
    input_width, input_height = args.input_width, args.input_height

    input_layer = tf.keras.layers.Input([input_height, input_width, 3])
    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, NUM_CLASS, i)
        bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, bbox_tensors)
    load_weights(model, args.yolo_weights)
    model_name = args.yolo_weights.split("/")[-1].replace(".weights", '')
    model.summary()
    os.makedirs(args.tf_models, exist_ok=True)
    tf.keras.models.save_model(model, f"{args.tf_models}/{model_name}.h5")
