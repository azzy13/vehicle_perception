import cv2
import numpy as np
from pycocotools.coco import COCO

import os

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class CARLADroneDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="coco.json",
        img_size=(608, 1088),
        preproc=None,
        eval=False
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "carla_drone_data")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, self.json_file))
        #self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.video_to_image = self.coco.dataset['video_to_image']
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.ids = self.generate_indices()
        self.annotations = self._load_coco_annotations()
        self.img_size = img_size
        self.preproc = preproc
        self.eval = eval

    def __len__(self):
        res = 0
        for entry in self.video_to_image:
            res += len(self.video_to_image[entry])
        return res
    
    def generate_indices(self):
        res = []
        for i in range(len(self.video_to_image)):
            res += self.video_to_image[str(i)]
        return res

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["id"]
        video_id = im_ann["video"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))
        track_ids = []

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]
            track_ids.append(obj["track_id"])

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        video_id = file_name.split("/")[-2]
        img_info = (height, width, int(frame_id), int(video_id), file_name, video_id.zfill(4))

        del im_ann, annotations

        return (res, img_info, file_name, track_ids)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name, track_ids = self.annotations[index]
        # load image and preprocess
        img_file = os.path.join(
            self.data_dir, file_name
        )
        img = cv2.imread(img_file)
        assert img is not None

        if self.eval:
            return img, res.copy(), img_info, np.array([id_]), track_ids
        else:
            return img, res.copy(), img_info, np.array([id_])

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
        if self.eval:
            img, target, img_info, img_id, track_ids = self.pull_item(index)
        else:
            img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        if self.eval:
            return img, target, img_info, int(img_id), track_ids
        return img, target, img_info, img_id
