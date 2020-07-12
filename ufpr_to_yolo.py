import os
import shutil
import pandas as pd
from tqdm import tqdm
import cv2

data_dir = "/data/UFPR-ALPR"
out_dir = "/data/UFPR-ALPR-yolo/"
os.makedirs(out_dir, exist_ok=True)


def convert(imgSize, box):
    xmin, ymin, xmax, ymax = box
    xcen = float((xmin + xmax)) / 2 / imgSize[1]
    ycen = float((ymin + ymax)) / 2 / imgSize[0]

    w = float((xmax - xmin)) / imgSize[1]
    h = float((ymax - ymin)) / imgSize[0]

    return xcen, ycen, w, h


for csv_file in ['training', 'validation', 'testing']:
    df = pd.read_csv(f"/data/csv/UFPR-ALPR/{csv_file}.csv")
    filenames = df.filename.values
    vehicle_bboxes = df.position_vehicle.values
    bboxes = df.position_plate.values

    yolofile = open(f"data/ufpr_alpr_{csv_file}_plate.txt", "w")


    for filename, bbox, vbbox in tqdm(zip(filenames, bboxes, vehicle_bboxes), total=len(filenames)):
        dir = "/".join(filename.split("/")[:-1])
        os.makedirs(f"{out_dir}/{dir}", exist_ok=True)
        # shutil.copy(f"{data_dir}/{filename}", f"{out_dir}/{filename}")

        vbbox = vbbox.split(" ")
        vbbox = [int(i) for i in vbbox]
        vbbox = [vbbox[0], vbbox[1], vbbox[0] + vbbox[2], vbbox[1] + vbbox[3]]

        bbox = bbox.split(" ")
        bbox = [int(i) for i in bbox]
        bbox = [bbox[0] - vbbox[0], bbox[1] - vbbox[1], bbox[0] + bbox[2] - vbbox[0], bbox[1] + bbox[3] - vbbox[1]]

        image = cv2.imread(f"{out_dir}/{filename}")
        image = image[vbbox[1]: vbbox[3], vbbox[0]: vbbox[2]]
        image_size = (image.shape[0], image.shape[1])  # (h, w)
        xcen, ycen, w, h = convert(image_size, bbox)

        cv2.imwrite(f"{out_dir}/{filename}", image)

        with open(f"{out_dir}/{filename}".replace(".png", ".txt"), 'w') as f:
            f.write(f"0 {xcen} {ycen} {w} {h}")

        yolofile.write(f"{out_dir}/{filename}" + "\n")

    yolofile.close()