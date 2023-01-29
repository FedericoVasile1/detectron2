import copy
import shutil
from collections import OrderedDict
import logging
import itertools
import os

import pycocotools.mask as mask_util
import numpy as np
import torch
import cv2
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import annotations_to_instances
import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode, Instances

from .metadatas import (
    INSTANCES
)
from .utils import mask_closest_center, get_centroid_coords, centroid_closest_to_img_ctr


class iHannesVideoEvaluator(DatasetEvaluator):
    """
    TODO
    """

    def __init__(
        self,
        dataset_name,
        eval_type,
        last_frame_id,
        task,
        save_option,
        test_set=None,
        distributed=False,
        output_dir=None
    ):
        assert dataset_name == "iHannesDataset" 
        assert save_option in ["wrong", "all", "none"]
        assert eval_type == "centroid"
        assert task in ["sem_seg", "inst_seg"]

        self._logger = logging.getLogger("detectron2")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self._eval_type = eval_type
        self._last_frame_id = last_frame_id
        self._task = task
        self._save_option = save_option
        self._test_set = test_set

        self._metadata = MetadataCatalog.get(dataset_name)
        self._classes_list = None
        if self._task == "inst_seg":
            self._classes_list = self._metadata.thing_classes
        elif self._task == "sem_seg":
            self._classes_list = self._metadata.stuff_classes
        else:
            raise Exception

        # dict where the key is the video identifier and the value is a dict 
        # where the key is the frame index and the value is the index of the 
        # class predicted for that frame
        self._videostring_2_framepreds = {}

    def reset(self):
        self._videostring_2_framepreds = {}

    def process(self, inputs, outputs):
        """
        Store here per-frame predictions. Per-video aggregation and accuracy 
        computation happen in evaluate()
        """
        keys = ["preshape_gt", "s_or_m"]
        if self._save_option is not None:
            keys += ["file_name", "height", "width"]

        for input, output in zip(inputs, outputs):
            assert isinstance(output, dict)

            frame_metadatas = {k: input[k] for k in keys}

            if (
                "-" in input["preshape_gt"] 
                or "-" in input["instance_name"] 
                or "-" in input["seq_num"]
            ):
                raise Exception
            video_string = "{}-{}-{}".format(
                input["instance_name"], input["preshape_gt"], input["seq_num"]
            )
            if video_string not in self._videostring_2_framepreds:
                self._videostring_2_framepreds[video_string] = {}

            if len(output["instances"]) == 0:
                # no instance predicted at the current frame
                frame_metadatas["idx_pred_cls"] = -1
                if self._save_option != "none":
                    # empty list since no instance is detected
                    frame_metadatas["instances"] = [] 
                image_id = int(input["image_id"])
                self._videostring_2_framepreds[video_string][image_id] = \
                    frame_metadatas
                continue

            # Get per-frame prediction using the method defined in eval_type
            if self._task == "inst_seg":
                if self._eval_type == "centroid":
                    frame_metadatas["idx_pred_cls"] = mask_closest_center(
                        output["instances"].pred_classes.to(self._cpu_device),
                        output["instances"].pred_masks.to(self._cpu_device)
                    )

                    if self._save_option != "none":
                        # save boxes, masks and predictios for visualization
                        instances = output["instances"].to(self._cpu_device)
                        # encode binary masks with rle to save memory 
                        frame_metadatas["instances"] = instances_to_coco_json(
                            instances, None
                        )
                        for f_m in frame_metadatas["instances"]:
                            # add the box mode used in instances_to_coco_json, 
                            # we will need it when decoding 
                            f_m["bbox_mode"] = BoxMode.XYWH_ABS

                else:
                    raise NotImplementedError

            elif self._task == "sem_seg":
                raise NotImplementedError

            image_id = int(input["image_id"])
            self._videostring_2_framepreds[video_string][image_id] = frame_metadatas

    def evaluate(self):
        # TODO
        #if self._distributed:
        #    comm.synchronize()
        #    frame_preds = comm.gather(self._video_preds, dst=0) 
        #    frame_preds = list(itertools.chain(*frame_preds))

        #    if not comm.is_main_process():
        #        return {}
        #else:
        #    frame_preds = self._frame_preds
        comm.synchronize()
        if comm.get_rank() > 0:
            return 

        if len(self._videostring_2_framepreds) == 0:
            self._logger.warning(
                "[iHannesVideoEvaluator] Did not receive valid predictions."
            )

        return self._eval_ihannesvideo() 

    def _eval_ihannesvideo(self):
        self._logger.info("Preparing results for iHannes video dataset")

        # preparing results to compute video accuracy
        cls_to_videopred = {cls: [] for cls in self._classes_list}
        cls_to_videopred = {s_m_a: copy.deepcopy(cls_to_videopred) 
                            for s_m_a in ["single", "multi", "all"]}
        for videostring, idxframe_to_metadatas in self._videostring_2_framepreds.items():
            idxs_frames = list(idxframe_to_metadatas.keys())
            # idxs must be from 0 to num-frames-in-video - 1
            assert len(set(idxs_frames)) == len(idxs_frames)
            assert np.min(idxs_frames) == 0 and np.max(idxs_frames) == (len(idxs_frames)-1)

            preds_frames = []
            # for each video, considers only the first _last_frame_id frames
            for i in idxs_frames:
                if i <= self._last_frame_id:
                    preds_frames.append(idxframe_to_metadatas[i]["idx_pred_cls"])
            # majority voting to obtain video prediction from per-frame predictions.
            # voting is performed without considering frames having no 
            # prediction (i.e., -1 value). However, if there is no prediction
            # along the whole video, the video prediction is no_prediction
            preds_frames = np.array(preds_frames)
            preds_frames = preds_frames[preds_frames != -1]
            if len(preds_frames):
                video_pred, _ = torch.mode(torch.tensor(preds_frames))
                video_pred = video_pred.item()
                cls = self._classes_list[video_pred]
            else:
                cls = "no_prediction"

            s_or_m = idxframe_to_metadatas[i]["s_or_m"]
            preshape_gt = idxframe_to_metadatas[i]["preshape_gt"]
            cls_to_videopred[s_or_m][preshape_gt].append(
                int(cls == preshape_gt)
            )
            cls_to_videopred["all"][preshape_gt].append(
                int(cls == preshape_gt)
            )

            if self._save_option == "all" \
                    or (self._save_option == "wrong" and cls != preshape_gt):
                # 0. create dir 
                video_dir = os.path.join(
                    self._output_dir, self._save_option+"-preds", videostring
                )
                if PathManager.isdir(video_dir):
                    shutil.rmtree(video_dir)
                PathManager.mkdirs(video_dir)

                idxs_frames.sort()
                # to keep track of the predictions along the frames of the video
                curvideo_cls_to_preds = {cls: 0 for cls in self._classes_list}
                curvideo_cls_to_preds["no_prediction"] = 0
                for i in idxs_frames:
                    if i > self._last_frame_id:
                        continue
                    # 1. load image and draw predicted masks
                    img = cv2.imread(idxframe_to_metadatas[i]["file_name"])
                    v = Visualizer(
                        img[:, :, ::-1], metadata=self._metadata, scale=1.0
                    )
                    # decode rle to binary masks
                    instances = annotations_to_instances(
                        idxframe_to_metadatas[i]["instances"], 
                        (idxframe_to_metadatas[i]["height"], idxframe_to_metadatas[i]["width"]),
                        "bitmask"
                    )
                    assert isinstance(instances, Instances)
                    # just to be clear, they are not ground truth but predicted,
                    # so fix it 
                    instances.set("pred_boxes", instances.gt_boxes)
                    instances.remove("gt_boxes")
                    instances.set("pred_classes", instances.gt_classes)
                    instances.remove("gt_classes")
                    if instances.has("gt_masks"):
                        instances.set("pred_masks", instances.gt_masks.tensor)
                        instances.remove("gt_masks")
                    instances.scores = torch.tensor(
                        [inst["score"] for inst in idxframe_to_metadatas[i]["instances"]]
                    )
                    img = v.draw_instance_predictions(instances)
                    img = copy.deepcopy(img.get_image()[:, :, ::-1])
                    # 2. color all mask centroid as red
                    if self._eval_type == "centroid":
                        centroids_list = []
                        if instances.has("pred_masks"):
                            centroids_list = get_centroid_coords(
                                instances.pred_masks,
                                idxframe_to_metadatas[i]["height"],
                                idxframe_to_metadatas[i]["width"]
                            )
                            for c in centroids_list:
                                x, y = c
                                cv2.circle(img, (x, y), 1, (0, 0, 255), 5)
                    else:
                        raise NotImplementedError
                    # 3. get closest-to-image-center centroid and color it as green 
                    if self._eval_type == "centroid":
                        x_y = centroid_closest_to_img_ctr(
                            centroids_list, 
                            idxframe_to_metadatas[i]["height"], 
                            idxframe_to_metadatas[i]["width"]
                        )
                        if x_y is not None:
                            cv2.circle(img, (x_y[0], x_y[1]), 1, (0, 255, 0), 5)
                    else:
                        raise NotImplementedError
                    # 4. print per-frame predictions so far on the current frame
                    cls = idxframe_to_metadatas[i]["idx_pred_cls"]
                    if cls == -1:
                        cls = "no_prediction"
                    else:
                        cls = self._classes_list[cls]
                    curvideo_cls_to_preds[cls] += 1 
                    sorted_by_value = {
                        k: v 
                        for k, v in sorted(curvideo_cls_to_preds.items(), 
                                           key=lambda item : item[1], 
                                           reverse=True)
                    }
                    s = "{}:{:>2d}/{:>2d}"
                    for idx, elem in enumerate(sorted_by_value.items()):
                        idx_cls, count = elem
                        if count == 0:
                            continue
                        text = s.format(idx_cls, count, i+1)
                        cv2.putText(
                            img, text, (0, 10+15*idx), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                            cv2.LINE_4
                        )
                    # 5. dump current frame
                    frame_name = str(i).zfill(8) + ".jpg"
                    frame_name = os.path.join(video_dir, frame_name)
                    cv2.imwrite(frame_name, img)

        # (OrderedDict[dict]): task_name -> {metric -> score}
        results = OrderedDict()
        for s_m_a in ["single", "multi", "all"]:
            self._logger.info("="*20)

            k = "{}-preshape objects".format(s_m_a)
            results[k] = {}

            total_acc = []
            for cls, preds_frames in cls_to_videopred[s_m_a].items():
                perclass_acc = (sum(preds_frames) / len(preds_frames) * 100) \
                                if len(preds_frames) != 0 else float("nan")
                self._logger.info(
                    "{:>6}-preshape {:>12} accuracy is {:.2f}% ({:2}/{:2})".format(
                        s_m_a.upper(), cls.upper(), perclass_acc, 
                        sum(preds_frames), len(preds_frames)
                    )
                )
                results[k][cls] = perclass_acc

                total_acc += preds_frames

            total_accuracy_frac = (sum(total_acc) / len(total_acc) * 100) \
                                   if len(total_acc) != 0 else float("nan")
            self._logger.info(
                "{:>6}-preshape {:>12} accuracy is {:.2f}% ({:2}/{:2})".format(
                    s_m_a.upper(), "TOTAL", total_accuracy_frac, 
                    sum(total_acc), len(total_acc)
                )
            )
            results[k]["total"] = total_accuracy_frac
        self._logger.info("="*20)

        return results