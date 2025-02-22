 # encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        #self.width = 0.5
        self.width = 1.5
        #self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        #self.input_size = (256, 512)
        #self.test_size = (256, 512)
        self.random_size = (18, 32)
        self.data_num_workers = 2
        self.max_epoch = 50
        self.print_interval = 20
        self.eval_interval = 10
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 5
        self.basic_lr_per_img = 0.00001
        self.warmup_epochs = 1
        self.drone_path = "carla_drone_data"
        self.kitti_path = "kitti"
        self.eval_path = "kitti"

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            CARLADroneDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )
        # Using carla, coco, kitti, roboflow, stanford cars, and ua-detrac, and bdd100k

        # Carla
        
        dataset = CARLADroneDataset(
            data_dir=os.path.join(get_yolox_datadir(), self.drone_path),
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )


        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )
        

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        return train_loader
    
    def get_eval_dataset(self):
        from yolox.data import KittiDataset, ValTransform

        valdataset = KittiDataset(
            data_dir=os.path.join(get_yolox_datadir(), self.kitti_path, "validation"),
            img_size=self.test_size,
            training=False,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        )

        return valdataset

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        valdataset = self.get_eval_dataset()

        '''
        if is_distributed:
            batch_size = 1
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        '''
        sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = 1
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import KittiEvaluator

        args = {}
        args["track_thresh"] = 0.77
        args["track_buffer"] = 300
        args["match_thresh"] = 0.67
        args["min_box_area"] = 10
        args["mot20"] = False
        args["ctra"] = False
        args = AttrDict(args)

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = KittiEvaluator(
            args=args,
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator