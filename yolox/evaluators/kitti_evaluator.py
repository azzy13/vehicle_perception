from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.ctra_bytetrack import CTRAByteTracker
from yolox.sort_tracker.sort import Sort
from yolox.deepsort_tracker.deepsort import DeepSort
from yolox.motdt_tracker.motdt_tracker import OnlineTracker

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import numpy as np

import json
import pandas as pd
import pandas
import os
import sys
from pycocotools.coco import COCO

def convert_multiple_to_coco(input_files, output_file):
    """
    Converts multiple tracking files into a single COCO JSON file.

    Args:
        input_files (list of str): List of paths to input tracking files.
        output_file (str): Path to save the combined COCO JSON file.
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "vehicle"}]  # Define "vehicle" as the object category
    }

    video_number = 0
    for input_file in input_files:
        # Load the current file into a dataframe
        data = pd.read_csv(input_file)#, names=['frame', 'id', 'x', 'y', 'w', 'h', '_1', '_2', '_3', '_4'])
        frame_ids = data['frame'].unique()

        for frame_id in frame_ids:
            # Add image metadata for each unique frame
            image_id = int(frame_id) + (100000 * video_number)
            coco_data["images"].append({
                "id": image_id,
                "file_name": f"frame_{int(frame_id)}.jpg",  # Assuming frame names are frame_<frame_id>.jpg
                "height": 0,  # Default height; replace with actual frame dimensions
                "width": 0   # Default width; replace with actual frame dimensions
            })

            # Extract rows for the current frame
            frame_data = data[data['frame'] == frame_id]
            for _, row in frame_data.iterrows():
                id = int(row["id"]) + (100000 * video_number)
                coco_data["annotations"].append({
                    "id": id,
                    "image_id": image_id,
                    "category_id": 1,  # Assuming a single category "vehicle"
                    "bbox": [row['x'], row['y'], row['w'], row['h']],
                    "area": row['w'] * row['h'],  # Area of the bounding box
                    "iscrowd": 0,  # Assuming all objects are non-crowded,
                    "score": row['score'] if 'score' in data.columns else 1  # Confidence score of the detection
                })
        video_number += 1

    # Save combined COCO JSON
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)



def write_results(filename, results, header=True):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w+') as f:
        if header:
            f.write("frame,id,x,y,w,h,score,-1,-1,-1\n")
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def write_results_no_score(filename, results, header=True):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    with open(filename, 'w+') as f:
        if header:
            f.write("frame,id,x,y,w,h,-1,-1,-1,-1\n")
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))

class KittiEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        decoder=None,
        result_folder="YOLOX_outputs/carla_drone/track_results",
        ground_truth_folder="YOLOX_outputs/carla_drone/ground_truth",
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        results_gt = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1
            
        tracker = None
        if self.args.ctra:
            tracker = CTRAByteTracker(self.args)
        else:
            tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        target_files = []
        gt_files = []
        logger.info("Total samples for evaluation: {}".format(n_samples))
        for cur_iter, (imgs, targets, info_imgs, ids, track_ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = info_imgs[5][0]
                self.args.track_buffer = 30
                self.args.track_thresh = ori_thresh

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 0:
                    if self.args.ctra:
                        tracker = CTRAByteTracker(self.args)
                    else:
                        tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                        result_filename_no_header = os.path.join(result_folder, '{}_nonheader.txt'.format(video_names[video_id]))
                        logger.info('save results to {}'.format(result_filename))
                        logger.info('save results to {}'.format(result_filename_no_header))
                        write_results(result_filename, results)
                        write_results(result_filename_no_header, results, header=False)
                        gt_filename = os.path.join(ground_truth_folder, '{}_gt.txt'.format(video_names[video_id]))
                        gt_filename_no_header = os.path.join(ground_truth_folder, '{}_nonheader.txt'.format(video_names[video_id]))
                        write_results_no_score(gt_filename, results_gt)
                        write_results_no_score(gt_filename_no_header, results_gt, header=False)
                        gt_files.append(gt_filename)
                        target_files.append(result_filename)
                        results = []
                        results_gt = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area:# and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))
                for i in range(targets.shape[1]):
                    line = targets[0][i][0:4]
                    line[2] = line[2] - line[0]
                    line[3] = line[3] - line[1]
                    track_id = track_ids[i].item()
                    results_gt.append((frame_id, [line.cpu().numpy().tolist()], [track_id]))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        convert_multiple_to_coco(target_files, os.path.join(result_folder, 'results.json'))
        convert_multiple_to_coco(gt_files, os.path.join(result_folder, 'gt.json'))
        eval_results = self.evaluate_prediction(os.path.join(result_folder, 'results.json'), os.path.join(result_folder, 'gt.json'), statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = 1
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, results, gt, statistics):
        try:
            if not is_main_process():
                return 0, 0, None

            logger.info("Evaluate in main process...")

            annType = ["segm", "bbox", "keypoints"]

            inference_time = statistics[0].item()
            track_time = statistics[1].item()
            n_samples = statistics[2].item()

            a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
            a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

            time_info = ", ".join(
                [
                    "Average {} time: {:.2f} ms".format(k, v)
                    for k, v in zip(
                        ["forward", "track", "inference"],
                        [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                    )
                ]
            )

            info = time_info + "\n"

            cocoGt = COCO(gt)
            cocoDt = COCO(results)
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info

        except Exception as e:
            logger.error("An error occurred during COCO evaluation: {}".format(e))
            return 0, 0, None
