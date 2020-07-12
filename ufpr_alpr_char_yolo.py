import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import os


chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

root_dir = "/data/UFPR-ALPR/"
out_dir = "/data/UFPR-ALPR-plate/"

def convert(imgSize, box):
    xmin, ymin, xmax, ymax = box
    xcen = float((xmin + xmax)) / 2 / imgSize[1]
    ycen = float((ymin + ymax)) / 2 / imgSize[0]

    w = float((xmax - xmin)) / imgSize[1]
    h = float((ymax - ymin)) / imgSize[0]

    return xcen, ycen, w, h


def char_detection(row, out_dir):
    filename = row.filename
    folders = "/".join(filename.split("/")[:-1])
    os.makedirs(f"{out_dir}/{folders}", exist_ok=True)
    plate = row.plate.replace('-', '')
    position_plate = row.position_plate
    position_plate = position_plate.split(" ")
    xmin, ymin, w, h = [int(x) for x in position_plate]

    image = cv2.imread(f"{root_dir}/{filename}")
    plate_image = image[ymin:ymin + h, xmin:xmin + w]
    cv2.imwrite(f"{out_dir}/{filename}", plate_image)

    image_size = plate_image.shape[:2]
    f = open(f"{out_dir}/{filename}".replace('.png', '.txt'), 'w')

    for i in range(7):
        char = row[f'char{i + 1}']
        char = char.split(" ")
        cxmin, cymin, cw, ch = [int(x) for x in char]
        cxmin -= xmin
        cymin -= ymin
        cxmin, cymin, cxmax, cxymax = cxmin, cymin, cxmin + cw, cymin + ch
        xcen, ycen, w, h = convert(image_size, (cxmin, cymin, cxmax, cxymax))
        plate_char = plate[i]
        plate_char = chars.index(plate_char)
        f.write(f"{plate_char} {xcen} {ycen} {w} {h} \n")
    f.close()

    return f"{out_dir}/{filename}"


for csv_file in ['training', 'validation', 'testing']:
    df = pd.read_csv(f"/data/csv/UFPR-ALPR/{csv_file}.csv")
    yolofile = open(f"data/ufpr_alpr_{csv_file}.txt", "w")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        outfilename = char_detection(row, out_dir)
        yolofile.write(outfilename + "\n")

    yolofile.close()