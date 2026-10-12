#!/bin/bash
#SBATCH -J sync_train               # Job name
#SBATCH -A ASC25022                 # Your allocation/project ID
#SBATCH -p gh                       # Grace‑Hopper GPU partition (H200 GPUs)
#SBATCH -N 16                        # DEFAULT: 1 node (override with sbatch -N)
#SBATCH -n 16                        # 1 task per job
#SBATCH -t 48:00:00                 # Max runtime is 48 hr per job on Vista
#SBATCH -o sbatch_logs/%j.log       # Stdout → log file
#SBATCH -e sbatch_logs/%j.log       # Stderr → log file
#SBATCH --verbose                   # Print directives as processed

# Exit on error
set -e

# Ensure working directory is submission directory
cd $SLURM_SUBMIT_DIR

# Create log directory for sbatch outputs
mkdir -p sbatch_logs

# Default values
RUN_NAME=""
RESUME=false
CONFIG="./configs/sync.yaml"

# Parse arguments: --run_name=<name> [--resume] [--config=<path>]
for arg in "$@"; do
  case $arg in
    --run_name=*)   RUN_NAME="${arg#*=}"; shift;;
    --resume)       RESUME=true; shift;;
    --config=*)     CONFIG="${arg#*=}"; shift;;
    *)
      echo "Unknown option $arg"
      exit 1;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  echo "Usage: sbatch $0 --run_name=<name> [--resume] [--config=<path>]"
  exit 1
fi

# Paths and directories
LOGDIR="/scratch/08659/jvoas/synch/checkpoints"
RUNLOGDIR="${LOGDIR}/${RUN_NAME}"
mkdir -p "$RUNLOGDIR"

DATASET_TARGET="dataset.ego_4d.Ego4D"
VIDS_PATH="/scratch/08659/jvoas/synch/datasets/ego4d-salt/Ego4D-reencoded/"
SPLITS_PATH="/scratch/08659/jvoas/synch/datasets/ego4d-salt/script/"
PRETRAINED_CKPT="/home1/Synchformer/logs/sync_models/ego4d/streaming/grid/baseline_streaming_synchformer/25-02-08T04-56-06/25-02-08T04-56-06_best_accuracy_1.pt"

# DDP setup
MASTER_ADDR=$(hostname -s)
if (( SLURM_JOB_NUM_NODES > 1 )); then
  WORKERS=$(scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER_ADDR)
fi
MASTER_PORT=$(comm -23 <(seq 49152 65535 | sort) \
               <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
export MASTER_ADDR MASTER_PORT

# GPU caching
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0

echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"
echo "MIOPEN cache dir: $MIOPEN_CUSTOM_CACHE_DIR"

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
conda activate synch

# Run
if [ "$RESUME" = true ]; then
  echo "Resuming run: $RUN_NAME"
  if [ -f "${RUNLOGDIR}/${RUN_NAME}_latest.pt" ]; then
    cp "${RUNLOGDIR}/${RUN_NAME}_latest.pt" "${RUNLOGDIR}/${RUN_NAME}.pt"
  fi
  srun python main.py \
    config="$RUNLOGDIR/cfg-$RUN_NAME.yaml" \
    training.resume=True training.finetune=False \
    data.dataset.target="$DATASET_TARGET" \
    data.vids_path="$VIDS_PATH" data.dataset.params.splits_path="$SPLITS_PATH" \
    logging.logdir="$LOGDIR" logging.run_name="$RUN_NAME" \
    training.base_batch_size=16 training.num_workers=7 training.patience=10
else
  echo "Starting new run: $RUN_NAME"
  srun python main.py \
    config="$CONFIG" \
    start_time="$(date +'%y-%m-%dT%H-%M-%S')" \
    logging.logdir="$LOGDIR" logging.run_name="$RUN_NAME" \
    data.dataset.target="$DATASET_TARGET" \
    data.vids_path="$VIDS_PATH" data.dataset.params.splits_path="$SPLITS_PATH" \
    model.pretrained_checkpoint="$PRETRAINED_CKPT" \
    training.base_batch_size=16 training.num_workers=7 training.patience=10
fi


# interactive_run.sh
#!/bin/bash
# Use this inside an idev session to launch on your reserved nodes

# Exit on error
set -e

# Ensure this is running under an active idev allocation
if [ -z "$SLURM_JOB_NODELIST" ]; then
  echo "Error: run this script inside an idev session"
  exit 1
fi

# Default values
RUN_NAME="dev"
RESUME=false
CONFIG="./configs/sync.yaml"

# Parse arguments
for arg in "$@"; do
  case $arg in
    --run_name=*)   RUN_NAME="${arg#*=}"; shift;;
    --resume)       RESUME=true; shift;;
    --config=*)     CONFIG="${arg#*=}"; shift;;
    *)
      echo "Unknown option $arg"
      exit 1;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  echo "Usage: $0 --run_name=<name> [--resume] [--config=<path>]"
  exit 1
fi

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
conda activate synch

# DDP setup
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(comm -23 <(seq 49152 65535 | sort) \
                    <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"

# Paths
LOGDIR="$SCRATCH/synch/checkpoints"
RUNLOGDIR="${LOGDIR}/${RUN_NAME}"
mkdir -p "$RUNLOGDIR"
DATASET_TARGET="dataset.ego_4d.Ego4D"
VIDS_PATH="$SCRATCH/synch/datasets/ego4d-salt/Ego4D-reencoded/"
SPLITS_PATH="$SCRATCH/synch/datasets/ego4d-salt/script/"
#PRETRAINED_CKPT="$HOME/Synchformer/logs/sync_models/ego4d/streaming/grid/baseline_streaming_synchformer/25-02-08T04-56-06/25-02-08T04-56-06_best_accuracy_1.pt"
PRETRAINED_CKPT="$HOME/Synchformer/logs/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt"

# Run
if [ "$RESUME" = true ]; then
  echo "Resuming run: $RUN_NAME"
  if [ -f "${RUNLOGDIR}/${RUN_NAME}_latest.pt" ]; then
    cp "${RUNLOGDIR}/${RUN_NAME}_latest.pt" "${RUNLOGDIR}/${RUN_NAME}.pt"
  fi
  srun python main.py \
    config="$RUNLOGDIR/cfg-$RUN_NAME.yaml" \
    training.resume=True training.finetune=False \
    data.dataset.target="$DATASET_TARGET" \
    data.vids_path="$VIDS_PATH" data.dataset.params.splits_path="$SPLITS_PATH" \
    logging.logdir="$LOGDIR" logging.run_name="$RUN_NAME" \
    training.base_batch_size=16 training.num_workers=7 training.patience=10
else
  echo "Starting new run: $RUN_NAME"
  srun python main.py \
    config="$CONFIG" \
    start_time="$(date +'%y-%m-%dT%H-%M-%S')" \
    logging.logdir="$LOGDIR/$RUN_NAME" logging.run_name="$RUN_NAME" \
    data.dataset.target="$DATASET_TARGET" \
    data.vids_path="$VIDS_PATH" data.dataset.params.splits_path="$SPLITS_PATH" \
    ckpt_path="$PRETRAINED_CKPT" \
    data.skip_ids_path="$HOME/Synchformer/skip_files.json" \
    training.base_batch_size=28 training.num_workers=5 training.patience=20
fi
