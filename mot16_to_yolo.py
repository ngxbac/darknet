import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob

columns = [
    'frame_number',
    'identity_number',
    'xmin', 'ymin', 'w', 'h',
    'conf_score',
    'class',
    'visibility'
]

# Pedestrian 1
# Person on vehicle 2
# Car 3
# Bicycle 4
# Motorbike 5
# Non motorized vehicle 6
# Static person 7
# Distractor 8
# Occluder 9
# Occluder on the ground 10
# Occluder full 11
# Reflection 12

classes = [
    'Pedestrian', 'Person_on_vehicle', 'Car',
    'Bicycle', 'Motorbike', 'Non_motorized_vehicle',
    'static_person', 'distractor', 'occluder',
    'occluder_on_the_ground', 'occluder_full', 'reflection'
]

cls_to_label = {}
label_to_cls = {}
for i, cls in enumerate(classes):
    cls_to_label[cls] = i + 1
    label_to_cls[i + 1] = cls


def convert(imgSize, box):
    xmin, ymin, xmax, ymax = box
    xcen = float((xmin + xmax)) / 2 / imgSize[1]
    ycen = float((ymin + ymax)) / 2 / imgSize[0]

    w = float((xmax - xmin)) / imgSize[1]
    h = float((ymax - ymin)) / imgSize[0]

    return xcen, ycen, w, h


def convert_to_yolo(video, prefix):
    with open(f'{video}/seqinfo.ini', 'r') as f:
        lines = f.readlines()
    print(lines)
    image_width = int(lines[-4].strip().split("=")[-1])
    image_height = int(lines[-3].strip().split("=")[-1])

    with open(f'{video}/gt/gt.txt', 'r') as f:
        lines = f.readlines()

    gt = []
    for line in lines:
        line = line.strip().split(',')
        line = [float(x) for x in line]
        gt.append(line)
    gt = np.asarray(gt)
    df = pd.DataFrame(gt, columns=columns)
    df = df[(df['class'] == 3) | (df['class'] == 4) | (df['class'] == 5)]
    print(len(df))
    filenames = []
    for frame_number, xmin, ymin, w, h, cls in zip(
            df.frame_number, df.xmin, df.ymin, df.w, df.h, df['class']
    ):
        frame_number = str(int(frame_number)).zfill(6)
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmin + w), int(ymin + h)
        xcen, ycen, w, h = convert((image_height, image_width), (xmin, ymin, xmax, ymax))
        with open(f"{video}/img1/{frame_number}.txt", 'w') as f:
            f.write(f"{int(cls - 3)} {xcen} {ycen} {w} {h}")

        filenames.append(f"{prefix}/{video}/img1/{frame_number}.jpg")
    return filenames

all_videos = glob.glob(f'/data/tracking/MOT16/train/*')
filenames = []
for video_path in all_videos:
    filenames += convert_to_yolo(video_path, prefix='')


valid_videos = [
    'MOT16-02', 'MOT16-13'
]

with open(f'data/mot16_train.txt', 'w') as f:
    for filename in filenames:
        if not any([video in filename for video in valid_videos]):
            f.write(filename + "\n")


with open(f'data/mot16_valid.txt', 'w') as f:
    for filename in filenames:
        if any([video in filename for video in valid_videos]):
            f.write(filename + "\n")

