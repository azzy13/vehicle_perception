from yolox.data import get_yolox_datadir
from yolox.data import (
    CarlaDataset,
    KittiDataset,
    UADataset,
    BDDDataset,
    TrainTransform
)
import os
import cv2
import numpy as np

def load_dataset(selection, name):
    if selection == "carla":
        return CarlaDataset(
            data_dir=os.path.join(get_yolox_datadir(), "carla"),
            name=name,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )
    elif selection == "coco":
        return CarlaDataset(
            data_dir=os.path.join(get_yolox_datadir(), "coco"),
            name=name,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )
    elif selection == "kitti":
        return KittiDataset(
            data_dir=os.path.join(get_yolox_datadir(), "kitti", name),
            training=True,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )
    elif selection == "stanford":
        return CarlaDataset(
            data_dir=os.path.join(get_yolox_datadir(), "stanford_cars"),
            name=name,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )
    elif selection == "ua":
        return UADataset(
            data_dir=os.path.join(get_yolox_datadir(), "ua-detrac"),
            name=name,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )
    elif selection == "bdd":
        return BDDDataset(
            data_dir=os.path.join(get_yolox_datadir(), "bdd100k"),
            name=name,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )
    return None

def draw_bboxes(dataset, index):
    result = dataset.pull_item(index)
    img = result[0]
    bboxes = result[1]
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        start_point = (x1, y1)
        end_point = (x2, y2)
        sub_img = img[y1:y2, x1:x2]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        img[y1:y2, x1:x2] = res

    return img

def save_image(image, dest):
    cv2.imwrite(dest, image) 