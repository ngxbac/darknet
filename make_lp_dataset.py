import pandas as pd
import numpy as np
import cv2
import ast
import os
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split


root_dir = "/data/CCPD/home/booy/booy/ccpd_dataset/"
out_dir = "./data/licesnse_plate/"
os.makedirs(out_dir, exist_ok=True)

def convert(imgSize, box):
    xmin, ymin, xmax, ymax = box
    xcen = float((xmin + xmax)) / 2 / imgSize[1]
    ycen = float((ymin + ymax)) / 2 / imgSize[0]

    w = float((xmax - xmin)) / imgSize[1]
    h = float((ymax - ymin)) / imgSize[0]

    return xcen, ycen, w, h


def create_dataset():
    df = pd.read_csv("/data/csv/CCPD/ccpd.csv.gz")
    df = df.sample(n=55000, replace=False)

    filenames = df.filename.values
    levels = df.level.values
    bboxes = df.bbox.values

    for filename, level, bbox in tqdm(zip(filenames, levels, bboxes), total=len(filenames)):
        text_name = filename.replace('.jpg', '.txt')
        image_path = os.path.join(root_dir, level, filename)
        image = cv2.imread(image_path)
        image_size = (image.shape[0], image.shape[1]) # (h, w)

        bbox = ast.literal_eval(bbox)
        xcen, ycen, w, h = convert(image_size, bbox)
        with open(f'{out_dir}/{text_name}', 'w') as f:
            f.write(f"{0} {xcen} {ycen} {w} {h}")

        cv2.imwrite(f"{out_dir}/{filename}", image)


def train_valid_split():
    text_files = glob.glob(f"{out_dir}/*.jpg")
    train_files, valid_files = train_test_split(text_files, test_size=0.1, random_state=42)
    train_files = [train_file[2:] for train_file in train_files]
    valid_files = [valid_file[2:] for valid_file in valid_files]
    # import pdb; pdb.set_trace()
    with open("./build/darknet/x64/data/license_plate_train.txt", "w") as f:
        for train_file in train_files:
            f.write(train_file + "\n")

    with open("./build/darknet/x64/data/license_plate_valid.txt", "w") as f:
        for valid_file in valid_files:
            f.write(valid_file + "\n")

if __name__ == '__main__':
    # create_dataset()
    train_valid_split()