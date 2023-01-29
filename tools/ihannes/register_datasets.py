import os
import json
import glob

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

from .get_ihannes_dataset import get_ihannesframes_dicts
from .metadatas import CLASSES_LIST
from projects.DeepLab.get_semseg_dataset import get_semseg_dicts


def register_datasets(cfg):
    datasets_base_path = os.path.join('datasets', 'synthetic', 'frames')

    task = None
    if "SemanticSegmentor" == cfg.MODEL.META_ARCHITECTURE:
        task = "sem_seg"
    elif "GeneralizedRCNN" == cfg.MODEL.META_ARCHITECTURE:
        task = "inst_seg"
    else:
        raise Exception

    for dataset_name in cfg.DATASETS.TRAIN:
        if dataset_name == "":
            continue

        if task == "inst_seg":
            dataset_path_imgs = os.path.join(
                datasets_base_path, dataset_name, 'RGB*'
            )
            dataset_path_imgs = glob.glob(dataset_path_imgs)
            if len(dataset_path_imgs) != 1:
                raise Exception
            dataset_path_imgs = dataset_path_imgs[0]
            coco_file = 'coco_train_preshape.json'
            dataset_path_coco = os.path.join(
                datasets_base_path, dataset_name, coco_file
            )
            register_coco_instances(
                dataset_name, {}, dataset_path_coco, dataset_path_imgs
            ) 

            with open(dataset_path_coco, "r") as f:
                cats = json.load(f)["categories"]
            thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
            assert thing_classes == CLASSES_LIST[task]
            assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == len(thing_classes)

        elif task == "sem_seg":
            DatasetCatalog.register(
                dataset_name,
                lambda elem=dataset_name : get_semseg_dicts(elem)
            )
            MetadataCatalog.get(dataset_name).set(
                stuff_classes=CLASSES_LIST[task], evaluator_type="sem_seg",
                ignore_label=[]
            )
            assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == len(CLASSES_LIST[task])

        else:
            raise Exception
        
    for dataset_name in cfg.DATASETS.TEST:
        if dataset_name == "":
            continue

        if dataset_name == "iHannesDataset":
            DatasetCatalog.register(
                dataset_name, 
                lambda elem=cfg.DATASETS.IHANNES.TEST_SET : get_ihannesframes_dicts(elem)
            )
            MetadataCatalog.get(dataset_name).set(evaluator_type="ihannes_video")

            if task == "inst_seg":
                MetadataCatalog.get(dataset_name).set(
                    thing_classes=CLASSES_LIST[task]
                )
            elif task == "sem_seg":
                MetadataCatalog.get(dataset_name).set(
                    stuff_classes=CLASSES_LIST[task], ignore_label=[]
                )
            else:
                raise Exception

            continue

        if task == "inst_seg":
            dataset_path_imgs = os.path.join(
                datasets_base_path, dataset_name, 'RGB*'
            )
            dataset_path_imgs = glob.glob(dataset_path_imgs)
            if len(dataset_path_imgs) != 1:
                raise Exception
            dataset_path_imgs = dataset_path_imgs[0]
            coco_file = None
            if dataset_name == 'origtextr-valset':
                coco_file = 'coco_val_preshape_downsampled.json'
            else:
                coco_file = 'coco_val_preshape.json'
            dataset_path_coco = os.path.join(
                datasets_base_path, dataset_name, 
                coco_file
            )
            register_coco_instances(
                dataset_name, {}, dataset_path_coco, dataset_path_imgs
            ) 

            with open(dataset_path_coco, "r") as f:
                cats = json.load(f)["categories"]
            thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
            assert thing_classes == CLASSES_LIST[task]
            assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == len(thing_classes)

        elif task == "sem_seg":
            DatasetCatalog.register(
                dataset_name,
                lambda elem=dataset_name : get_semseg_dicts(elem)
            )
            MetadataCatalog.get(dataset_name).set(
                stuff_classes=CLASSES_LIST[task], evaluator_type="sem_seg",
                ignore_label=[]
            )
            assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == len(CLASSES_LIST[task])

        else:
            raise Exception
