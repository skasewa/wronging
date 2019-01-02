# Generates corruptions through temeperature sampling (TS).
# This script only generated one corrupted version; however, it can be easily modified to generate multiple versions.

export HOMEDIR=/home/user/wronging
export MODEL_DIR=${HOMEDIR}/models/s2s-models
export TEST_SOURCES=${HOMEDIR}/data/train/fce/targets.txt   # these correct sentences will be corrupted
export TEST_PREDS_DIR=${HOMEDIR}/data/NMT
mkdir -p $TEST_PREDS_DIR
export TEST_PREDS_PREFIX=${TEST_PREDS_DIR}/fce-ts   # fce-ts-$TEMP-$MODEL.txt is the file that will be written
export TEMP=0.05

export MODEL=50000
export TEST_PREDS=${TEST_PREDS_PREFIX}-${TEMP}-${MODEL}
python3 -m bin.infer \
  --tasks "
    - class: DecodeText
      params:
        unk_replace: True" \
  --model_dir $MODEL_DIR \
  --checkpoint_path ${MODEL_DIR}/model.ckpt-$MODEL \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TEST_SOURCES" \
  --sampler "multinomial" \
  --temp ${TEMP} \
  >  ${TEST_PREDS}.txt
 
echo "Predictions $TEST_PREDS written from Model $MODEL"

