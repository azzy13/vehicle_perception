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
        preproc=None,
    ):
        super().__init__(img_size)
        self.data_dir = os.path.join(data_dir, "content", "UA-DETRAC", "DETRAC_Upload")
        self.image_dir = os.path.join(self.data_dir, "images", name)
        self.label_dir = os.path.join(self.data_dir, "labels", name)

        self.annotations = self._load_annotations()
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return self.img_count
    
    def _load_annotations(self):
        self.img_count = 0
        annos = []
        annotation_files = glob.glob(os.path.join(self.label_dir, "*.txt"))
        annotation_files.sort()
        for index in range(len(annotation_files)):
            annotation_file = annotation_files[index]
            image_name = (((annotation_file.split("/"))[-1]).split(".")[0]) + ".jpg"
            with open(annotation_file, "r") as f:
                lines = f.readlines()
                anno = {}
                anno["frame_path"] = image_name
                anno["frame_id"] = self.img_count
                anno["objs"] = []
                for line in lines:
                    items = line.split(" ")
                    if (len(items) == 5):
                        obj = {}
                        obj["bbox"] = []
                        obj["bbox"].append(float(items[1]))
                        obj["bbox"].append(float(items[2]))
                        obj["bbox"].append(float(items[3]))
                        obj["bbox"].append(float(items[4]))
                        anno["objs"].append(obj)
                annos.append(anno)
            self.img_count += 1
        return annos
    
    def pull_item(self, index):
        im_ann = self.annotations[index]
        file_name = os.path.join(self.image_dir, im_ann["frame_path"])

        # load image and preprocess
        img = cv2.imread(file_name)
        assert img is not None

        width = img.shape[1]
        height = img.shape[0]

        annotations = im_ann["objs"]
        frame_id = im_ann["frame_id"]
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0] * width
            y1 = obj["bbox"][1] * height
            x2 = obj["bbox"][2] * width
            y2 = obj["bbox"][3] * height
            if x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2 - x1, y2 - y1]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = 0

        
        img_info = (height, width, frame_id, file_name)

        return img, res.copy(), img_info, index

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
        img, target, img_info, img_id = self.pull_item(index)
        initial_size = img.shape

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, int(img_id)
