# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# data folder
DATASET_ROOT='./data/iris/datasets/fipt/indoor_synthetic/'
DATASET='synthetic'
# scene name
SCENE='bedroom_mi'
LDR_IMG_DIR='Image'
EXP='fipt_syn_bedroom_mi'
VAL_FRAME=7
CRF_BASIS=3
# whether has part segmentation
HAS_PART=1
SPP=256
spp=256
SPLIT=${SPLIT:-train}
RELIGHT_SPLIT=${RELIGHT_SPLIT:-relight}
RUN_RELIGHT=${RUN_RELIGHT:-0}

python render.py --experiment_name $EXP --device 0 --ckpt last_1.ckpt \
  --dataset $DATASET $DATASET_ROOT$SCENE \
  --emitter_path checkpoints/$EXP/bake --output_path 'outputs/'$EXP'/output' \
  --split $SPLIT --ldr_img_dir $LDR_IMG_DIR \
  --SPP $SPP --spp $spp --crf_basis $CRF_BASIS

python render_video.py --experiment_name $EXP --device 0 --ckpt last_1.ckpt \
  --dataset $DATASET $DATASET_ROOT$SCENE --emitter_path checkpoints/$EXP/bake --output_path 'outputs/'$EXP'/video' \
  --split $SPLIT --ldr_img_dir $LDR_IMG_DIR \
  --SPP $SPP --spp $spp --crf_basis $CRF_BASIS

if [[ "${RUN_RELIGHT}" == "1" ]]; then
  python render_relight.py --experiment_name $EXP --device 0 --ckpt last_1.ckpt --mode traj --dataset $DATASET $DATASET_ROOT$SCENE --emitter_path checkpoints/$EXP/bake --output_path 'outputs/'$EXP'/relight/video_relight_0' \
    --split $RELIGHT_SPLIT --ldr_img_dir $LDR_IMG_DIR \
    --light_cfg 'configs/fipt/bedroom/relight_0.yaml' \
    --SPP $SPP --spp $spp --crf_basis $CRF_BASIS
fi
