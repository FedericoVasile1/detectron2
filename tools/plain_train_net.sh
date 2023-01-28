python plain_train_net.py \
--config-file '../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml' \
DATASETS.TRAIN "('sphere-ctr-origtextr-trainset','sphere-out-ctr-origtextr-trainset',)" \
DATASETS.TEST "('iHannesDataset','origtextr-valset','sphere-origtextr-valset',)" \
MODEL.WEIGHTS 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl' \
INPUT.MASK_FORMAT bitmask \
SOLVER.IMS_PER_BATCH 8 SOLVER.STEPS '()' SOLVER.MAX_ITER 40000 \
SOLVER.BASE_LR 0.00025 TEST.EVAL_PERIOD 20 \
MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 512 MODEL.ROI_HEADS.NUM_CLASSES 4 \
INPUT.BLUR.ENABLED True \
OUTPUT_DIR ./output-sphere
