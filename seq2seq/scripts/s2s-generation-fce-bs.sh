# Generates corruptions through beam search (BS).
# This generates a single file with all the beams of any particular output chunked together.
# See tsvutils.py for wrappers to convert this file into a usable tsv file

export HOMEDIR=/home/user/wronging
export MODEL_DIR=${HOMEDIR}/models/s2s-models
export TEST_SOURCES=${HOMEDIR}/data/train/fce/targets.txt   # these correct sentences will be corrupted
export TEST_PREDS_DIR=${HOMEDIR}/data/NMT
mkdir -p $TEST_PREDS_DIR
export TEST_PREDS_PREFIX=${TEST_PREDS_DIR}/fce-bs   # fce-bs-$BEAMS-$MODEL.txt is the file that will be written
export BEAMS=11

export MODEL=50000
export TEST_PREDS=${TEST_PREDS_PREFIX}-${BEAMS}-${MODEL}
python3 -m bin.infer \
  --tasks "
    - class: DecodeText
      params:
        unk_replace: True" \
  --model_params "
    inference.beam_search.beam_width: ${BEAMS}" \
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

