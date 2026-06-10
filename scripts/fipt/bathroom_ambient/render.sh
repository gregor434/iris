# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# data folder
DATASET_ROOT='./data/iris/datasets/fipt/indoor_synthetic/'
DATASET='synthetic'
# scene name
SCENE='bathroom_mi'
LDR_IMG_DIR='Image'
EXP='fipt_syn_bathroom_mi'
VAL_FRAME=10
CRF_BASIS=3
# whether has part segmentation
HAS_PART=1
SPP=32
spp=32
RENDER_CHUNK_SIZE=8192
SPLIT=${SPLIT:-val}

python render.py --experiment_name $EXP --device 0 --ckpt last_1.ckpt \
  --dataset $DATASET $DATASET_ROOT$SCENE \
  --emitter_path checkpoints/$EXP/bake --output_path 'outputs/'$EXP'/output' \
  --split $SPLIT --ldr_img_dir $LDR_IMG_DIR \
  --SPP $SPP --spp $spp --render_chunk_size $RENDER_CHUNK_SIZE --crf_basis $CRF_BASIS

python render_video.py --experiment_name $EXP --device 0 --ckpt last_1.ckpt \
  --dataset $DATASET $DATASET_ROOT$SCENE --emitter_path checkpoints/$EXP/bake --output_path 'outputs/'$EXP'/video' \
  --split $SPLIT --ldr_img_dir $LDR_IMG_DIR \
  --SPP $SPP --spp $spp --render_chunk_size $RENDER_CHUNK_SIZE --crf_basis $CRF_BASIS
