python train_net.py \
--config-file 'configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml' \
DATASETS.TRAIN "('sphere-semseg-ctr-trainset',)" \
DATASETS.TEST "('sphere-semseg-ctr-valset',)" \
SOLVER.MAX_ITER 100000 \
TEST.EVAL_PERIOD 2500 \
SOLVER.CHECKPOINT_PERIOD 2500 \
MODEL.SEM_SEG_HEAD.NUM_CLASSES 6 \
SOLVER.IMS_PER_BATCH 32 \
DATALOADER.NUM_WORKERS 10 \
INPUT.CROP.ENABLED True \
INPUT.CROP.SIZE "(432, 576)" \
INPUT.MIN_SIZE_TRAIN '(480, 512, 530, 560, 600, 620)' \
INPUT.MAX_SIZE_TRAIN 1024 \
INPUT.MIN_SIZE_TEST 432 \
INPUT.MAX_SIZE_TEST 1024 \
OUTPUT_DIR ./output-sphere-semseg-ctr
