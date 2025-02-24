# encoding: utf-8
import os
import torch
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
        # Evaluate only vehicles
        self.num_classes = 1  
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.data_num_workers = 2
        self.test_conf = 0.1
        self.nmsthre = 0.7
        # Root path for BDD100k under your YOLOX dataset directory
        self.bdd_path = "bdd100k"  

    def get_eval_dataset(self):
        """
        Load BDD100k validation dataset from:
        <your_yolox_data_dir>/bdd100k/images/100k/val
        <your_yolox_data_dir>/bdd100k/labels/det_val.json
        """
        from yolox.data import BDDDataset, ValTransform

        val_dataset = BDDDataset(
            data_dir=os.path.join(get_yolox_datadir(), self.bdd_path),
            name='val',  # This typically points to images/100k/val
            img_size=self.test_size,
            training=False,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )
        return val_dataset

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        """
        Create an evaluation DataLoader for the BDD100k val set.
        """
        val_dataset = self.get_eval_dataset()
        sampler = torch.utils.data.SequentialSampler(val_dataset)
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": 1,  # for evaluation, typically 1 or small batch size
        }
        val_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)
        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        """
        Instantiate the BDDEvaluator (ensure you have a BDDEvaluator in your codebase).
        """
        from yolox.evaluators import BDDEvaluator
        args = {
            "track_thresh": 0.77,
            "track_buffer": 300,
            "match_thresh": 0.67,
            "min_box_area": 10,
            "mot20": False,
            "ctra": False,
        }
        args = AttrDict(args)
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = BDDEvaluator(
            args=args,
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
