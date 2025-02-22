import cv2
import numpy as np
import glob
import torch

import os
import json

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class UADataset(Dataset):
    def __init__(
        self,
        data_dir,
        name='train',
        img_size=(608, 1088),
        training=False,
        preproc=None,
    ):
        super().__init__(img_size)
        self.data_dir = os.path.join(data_dir, "content", "UA-DETRAC", "DETRAC_Upload")
        self.image_dir = os.path.join(self.data_dir, "images", name)
        self.label_dir = os.path.join(self.data_dir, "labels", name)

        self.annotations, self.ids_ = self._load_annotations()
        self.img_size = img_size
        self.preproc = preproc
        self.train_mode = training

    def __len__(self):
        return len(self.ids_)

    def _load_annotations(self):
        ids = []
        annos = {}
        annotation_files = glob.glob(os.path.join(self.label_dir, "*.txt"))
        annotation_files.sort()

        img_count = 0
        video_count = 0
        for annotation_file in annotation_files:
            image_name = os.path.basename(annotation_file).replace(".txt", ".jpg")
            frame_annotations = []

            with open(annotation_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split(" ")
                    if len(items) == 5:  # YOLO format
                        obj = {
                            "bbox": [
                                float(items[1]),
                                float(items[2]),
                                float(items[3]),
                                float(items[4]),
                            ],
                            "class_id": int(items[0]),
                            "track_id": img_count,
                        }
                        frame_annotations.append(obj)

            annos[img_count] = {
                "frame_path": image_name,
                "objs": frame_annotations,
            }
            ids.append([video_count, img_count])
            img_count += 1

        return annos, ids

    def load_anno_from_id(self, id_):
        im_ann = self.annotations[id_]
        width, height = 1088, 608
        annotations = im_ann["objs"]

        objs = []
        track_ids = []
        for obj in annotations:
            x1 = (obj["bbox"][0] - obj["bbox"][2] / 2) * width
            y1 = (obj["bbox"][1] - obj["bbox"][3] / 2) * height
            x2 = x1 + obj["bbox"][2] * width
            y2 = y1 + obj["bbox"][3] * height

            if x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)
                track_ids.append(obj["track_id"])

        num_objs = len(objs)
        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = obj["class_id"]

        file_name = os.path.join(self.image_dir, im_ann["frame_path"])
        img_info = (height, width, id_, file_name)

        return res, img_info, file_name, track_ids

    def pull_item(self, index):
        id_ = self.ids_[index]
        res, img_info, file_name, track_ids = self.load_anno_from_id(id_)
        img = cv2.imread(file_name)
        assert img is not None, f"Image not found: {file_name}"

        if self.train_mode:
            return img, res.copy(), img_info, np.array([id_[1]])
        return img, res.copy(), img_info, np.array([id_[1]]), track_ids

    @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id, track_ids = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, int(img_id), track_ids
