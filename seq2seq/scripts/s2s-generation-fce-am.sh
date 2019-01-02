# Generates corruptions through argmax sampling (AM).
# Since argmax is deterministic, a loop over different models is required to generate multiple corrupted versions

export HOMEDIR=/home/user/wronging
export MODEL_DIR=${HOMEDIR}/models/s2s-models
export TEST_SOURCES=${HOMEDIR}/data/train/fce/targets.txt   # these correct sentences will be corrupted
export TEST_PREDS_DIR=${HOMEDIR}/data/NMT
mkdir -p $TEST_PREDS_DIR
export TEST_PREDS_PREFIX=${TEST_PREDS_DIR}/fce-am   # fce-am-$MODEL.txt is the file that will be written

for i in 50000 51000 52000
do

  export MODEL=$i
  export TEST_PREDS_PREFIX=${TEST_PREDS}-${MODEL}
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
    --sampler "argmax" \
    --temp 0.000000000000000 \
    >  ${TEST_PREDS}.txt
  
  echo "Predictions $TEST_PREDS written from Model $MODEL"

done

