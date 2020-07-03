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
    ANCHORS = [5, 32,  10, 21,   8, 34,  13, 28,  13, 32,  15, 30,  22, 20,  15, 35,  16, 36]
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
