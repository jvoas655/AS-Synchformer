#!/bin/bash
# interactive_run.sh — launch training with torchrun on a single-node multi-GPU server with AWS defines

set -e

# defaults
RUN_NAME="dev"
RESUME=false
VALIDATE=false
CONFIG="./configs/stream_synchformer_v2.yaml"
RESUME_TIME=""

# parse args
for arg in "$@"; do
  case $arg in
    --run_name=*)   RUN_NAME="${arg#*=}" ;;
    --resume)       RESUME=true ;;
    --validate)       VALIDATE=true ;;
    --config=*)     CONFIG="${arg#*=}" ;;
    --resume_time=*) RESUME_TIME="${arg#*=}" ;;
    *) echo "Unknown option $arg"; exit 1 ;;
  esac
done

echo "==> Run name:    $RUN_NAME"
echo "==> Resume:      $RESUME"
echo "==> Config file: $CONFIG"

# DDP setup
export MASTER_ADDR=$(hostname -s)
# pick a free port between 49152–65535
export MASTER_PORT=$(comm -23 <(seq 49152 65535 | sort) \
  <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n1)
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

export OMP_NUM_THREADS=2

# detect number of GPUs available
NGPU=$(python - <<'PYCODE'
import torch
n = torch.cuda.device_count()
print(n if n>0 else 1)
PYCODE
)
echo "Found $NGPU GPUs; using torchrun with --nproc_per_node=$NGPU"

# paths
LOGDIR="logs/sync_models/checkpoints"
RUNLOGDIR="$LOGDIR/$RUN_NAME"
mkdir -p "$RUNLOGDIR"

PRETRAINED_CKPT="logs/sync_models/ego4d/streaming/grid/stream_synchformer_v2/25-04-25T01-38-30/25-04-25T01-38-30_best_accuracy_1.pt"
# choose launcher
LAUNCHER="torchrun --nnodes=1 --nproc_per_node=$NGPU --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"

# launch training
if [ "$RESUME" = true ]; then
  echo "==> Resuming run: $RUN_NAME"
  if [ -f "$RUNLOGDIR/${RUN_NAME}_latest.pt" ]; then
    cp "$RUNLOGDIR/${RUN_NAME}_latest.pt" "$RUNLOGDIR/${RUN_NAME}.pt"
  fi

  $LAUNCHER python main.py \
    config="$CONFIG" \
    training.resume=True training.finetune=False \
    start_time="$RESUME_TIME" \
    logging.logdir="$LOGDIR/$RUN_NAME" logging.run_name="$RUN_NAME" \
    data.dataset.target="$DATASET_TARGET" \
    data.vids_path="$VIDS_PATH" \
    data.dataset.params.splits_path="$SPLITS_PATH" \
    ckpt_path="$LOGDIR/${RUN_NAME}/$RESUME_TIME/${RESUME_TIME}_best_accuracy_1.pt" \
    data.skip_ids_path="$HOME/Synchformer/skip_files.json" \
    training.base_batch_size=10 training.num_workers=5 training.patience=20

else
  echo "==> Starting new run: $RUN_NAME"

  
    if [ "$VALIDATE" = true ]; then
        $LAUNCHER main.py \
            config="$CONFIG" \
            start_time="$(date +'%y-%m-%dT%H-%M-%S')" \
            logging.logdir="$LOGDIR/$RUN_NAME" logging.run_name="$RUN_NAME" \
            ckpt_path="$PRETRAINED_CKPT" \
            data.skip_ids_path="$HOME/Synchformer/skip_files.json" \
            training.startup_validate=True training.validate_only=True \
            data.dataset.params.size_ratio.valie=0.2 \
            logging.log_frequency=100 logging.val_frequency=2500 \
            training.acum_steps=4 \
            training.base_batch_size=56 training.num_workers=2 training.patience=20
    else
        $LAUNCHER main.py \
            config="$CONFIG" \
            start_time="$(date +'%y-%m-%dT%H-%M-%S')" \
            logging.logdir="$LOGDIR/$RUN_NAME" logging.run_name="$RUN_NAME" \
            ckpt_path="$PRETRAINED_CKPT" \
            data.skip_ids_path="$HOME/Synchformer/skip_files.json" \
            data.dataset.params.size_ratio.valid=0.2 \
            training.acum_steps=2 \
            training.base_batch_size=6 training.num_workers=2 training.patience=20
    fi
fi
