export HOMEDIR=/home/user/wronging

export VOCAB_SOURCE=${HOMEDIR}/data/train/fce/vocab.txt
export VOCAB_TARGET=${HOMEDIR}/data/train/fce/vocab.txt
export TRAIN_SOURCES=${HOMEDIR}/data/train/fce/targets.txt   # by convention, targets.txt contains correct sentences ; this is the SOURCE in NMT training
export TRAIN_TARGETS=${HOMEDIR}/data/train/fce/sources.txt   # sources.txt contains sentences with errors ; this is the TARGET in NMT training
export DEV_SOURCES=${HOMEDIR}/data/dev/fce/targets.txt
export DEV_TARGETS=${HOMEDIR}/data/dev/fce/sources.txt

export TRAIN_STEPS=100000

export MODEL_DIR=${HOMEDIR}/models/s2s-models
mkdir -p $MODEL_DIR

python3 -m bin.train \
  --config_paths="./example_configs/s2s-1layer-model.yml,
                  ./example_configs/s2s-fce-train.yml,
                  ./example_configs/s2s-fce-metrics.yml"\
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS " \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES
      target_files:
        - $DEV_TARGETS" \
  --batch_size 128 \
  --save_checkpoints_steps 5000 \
  --keep_checkpoint_max 0 \
  --keep_checkpoint_every_n_hours 2 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR

