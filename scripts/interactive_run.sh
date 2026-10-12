#!/bin/bash
# interactive_run.sh — launch training on your idev‑allocated nodes via srun

set -e

# ensure we're on compute nodes from an idev session
if [ -z "$SLURM_JOB_NODELIST" ]; then
  echo "Error: Must be run inside an idev session"
  exit 1
fi

# defaults
RUN_NAME="dev"
RESUME=false
CONFIG="./configs/stream_synchformer_v2.yaml"
RESUME_TIME=""
VALIDATE=false

# parse args
for arg in "$@"; do
  case $arg in
    --run_name=*)   RUN_NAME="${arg#*=}" ;;
    --resume)       RESUME=true ;;
    --validate)       VALIDATE=true ;;
    --config=*)     CONFIG="${arg#*=}" ;;
    --resume_time=*)     RESUME_TIME="${arg#*=}" ;;
    *) echo "Unknown option $arg"; exit 1 ;;
  esac
done

echo "==> Run name:    $RUN_NAME"
echo "==> Resume:      $RESUME"
echo "==> Config file: $CONFIG"

# load compiler & CUDA
module load gcc/13.2.0 cuda/12.5 nvmath/12.5.0

# activate your Conda environment installed in $WORK
if [ -f "$WORK/vista/etc/profile.d/conda.sh" ]; then
  source "$WORK/vista/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: Cannot find your conda installation under \$WORK/vista"
  exit 1
fi
source "$HOME/start_env.sh"
export OMP_NUM_THREADS=12
conda activate synch

# DDP setup
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(comm -23 <(seq 49152 65535 | sort) \
  <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n1)
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

# paths
LOGDIR="$SCRATCH/synch/checkpoints"
RUNLOGDIR="$LOGDIR/$RUN_NAME"
mkdir -p "$RUNLOGDIR"

DATASET_TARGET="dataset.ego_4d.Ego4D"
VIDS_PATH="$WORK/synch/datasets/ego4d-salt/Ego4D-reencoded/"
SPLITS_PATH="$WORK/synch/datasets/ego4d-salt/script/"
#PRETRAINED_CKPT="$HOME/Synchformer/logs/sync_models/ego4d/streaming/grid/baseline_streaming_synchformer/25-02-08T04-56-06/25-02-08T04-56-06_best_accuracy_1.pt"
#PRETRAINED_CKPT="$HOME/Synchformer/logs/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt"
#PRETRAINED_CKPT="/scratch/08659/jvoas/synch/checkpoints/stream_sync_full_train_rev4/25-04-24T10-20-00/25-04-24T10-20-00_best_accuracy_1.pt"/scratch/08659/jvoas/synch/checkpoints/stream_sync_full_train_rev4/25-04-24T16-53-14/25-04-24T16-53-14_best_accuracy_1.pt
PRETRAINED_CKPT="/scratch/08659/jvoas/synch/checkpoints/stream_sync_mod_full_train/25-04-25T01-38-30/25-04-25T01-38-30_best_accuracy_1.pt"
#PRETRAINED_CKPT="$HOME/Synchformer/logs/sync_models/ego4d/streaming/grid/stream_synchformer_v2_ft/25-04-26T11-20-35_best_accuracy_1.pt"

# launch training
if [ "$RESUME" = true ]; then
  echo "==> Resuming run: $RUN_NAME"
  if [ -f "$RUNLOGDIR/${RUN_NAME}_latest.pt" ]; then
    cp "$RUNLOGDIR/${RUN_NAME}_latest.pt" "$RUNLOGDIR/${RUN_NAME}.pt"
  fi

  srun python main.py \
    config="$CONFIG" \
    training.resume=True training.finetune=False \
    start_time="$RESUME_TIME" \
    logging.logdir="$LOGDIR/$RUN_NAME" logging.run_name="$RUN_NAME" \
    data.dataset.target="$DATASET_TARGET" \
    data.vids_path="$VIDS_PATH" \
    data.dataset.params.splits_path="$SPLITS_PATH" \
    ckpt_path="$LOGDIR/${RUN_NAME}/$RESUME_TIME/${RESUME_TIME}_best_accuracy_1.pt" \
    data.skip_ids_path="$HOME/Synchformer/skip_files.json" \
    training.base_batch_size=6 training.num_workers=5 training.patience=20

else
  echo "==> Starting new run: $RUN_NAME"
  if [ "$VALIDATE" = true ]; then
    srun python main.py \
        config="$CONFIG" \
        start_time="$(date +'%y-%m-%dT%H-%M-%S')" \
        logging.logdir="$LOGDIR/$RUN_NAME" logging.run_name="$RUN_NAME" \
        data.dataset.target="$DATASET_TARGET" \
        data.vids_path="$VIDS_PATH" \
        data.dataset.params.splits_path="$SPLITS_PATH" \
        ckpt_path="$PRETRAINED_CKPT" \
        data.skip_ids_path="$HOME/Synchformer/skip_files.json" \
        training.startup_validate=True training.validate_only=True \
        training.base_batch_size=6 training.num_workers=8 training.patience=20
  else
    srun python main.py \
        config="$CONFIG" \
        start_time="$(date +'%y-%m-%dT%H-%M-%S')" \
        logging.logdir="$LOGDIR/$RUN_NAME" logging.run_name="$RUN_NAME" \
        data.dataset.target="$DATASET_TARGET" \
        data.vids_path="$VIDS_PATH" \
        data.dataset.params.splits_path="$SPLITS_PATH" \
        ckpt_path="$PRETRAINED_CKPT" \
        data.skip_ids_path="$HOME/Synchformer/skip_files.json" \
        training.base_batch_size=6 training.num_workers=8 training.patience=20
  fi
fi
