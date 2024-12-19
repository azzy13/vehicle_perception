import cv2
import numpy as np
import glob
import torch

import os
import json

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class KittiDataset(Dataset):
    def __init__(
        self,
        data_dir="/isis/home/hasana3/ByteTrack/datasets/kitti/training",
        img_size=(608, 1088),
        training=False,
        preproc=None,
    ):
        super().__init__(img_size)
        self.data_dir = data_dir

        self.annotations, self.objs, self.ids_ = self._load_annotations()
        self.img_size = img_size
        self.preproc = preproc
        self.training = training

    def __len__(self):
        return self.img_count
    
    def _load_annotations(self):
        self.img_count = 0
        self.obj_count = 0
        annos = {}
        objs = []
        ids = []
        json_files = glob.glob(os.path.join(self.data_dir, "label_02_json", "*.json"))
        json_files.sort()
        for json_file in json_files:
            with open(json_file, "r") as f:
                video_index = int(json_file.split("/")[-1].split(".")[0])
                annos[video_index] = json.load(f)
                for key in annos[video_index].keys():
                    objs.append(annos[video_index][key]["objs"])
                    self.obj_count += len(annos[video_index][key]["objs"])
                    ids.append([video_index, self.img_count + int(key), key])
                self.img_count += len(annos[video_index])
        return annos, objs, ids

    def load_anno_from_id(self, id_):
        im_ann = self.annotations[id_[0]][id_[2]]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = id_[2]
        annotations = im_ann["objs"]
        objs = []
        track_ids = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = obj["bbox"][2]
            y2 = obj["bbox"][3]
            if x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2 - x1, y2 - y1]
                objs.append(obj)
                track_ids.append(obj["track_id"])

        num_objs = len(objs)
        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = 0

        file_name = os.path.join(self.data_dir, im_ann["frame_path"])
        video_id = file_name.split("/")[-2]
        img_info = (height, width, int(frame_id), int(video_id), file_name, video_id.zfill(4))
        del im_ann, annotations

        return (res, img_info, file_name, track_ids)
    
    def pull_item(self, index):
        id_ = self.ids_[index]
        res, img_info, file_name, track_ids = self.load_anno_from_id(id_)
        # load image and preprocess
        img = cv2.imread(file_name)
        assert img is not None
        return img, res.copy(), img_info, np.array([id_[1]]), track_ids

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id, track_ids = self.pull_item(index)
        initial_size = img.shape

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        if self.training:
            return img, target, img_info, int(img_id)
        return img, target, img_info, int(img_id), track_ids
