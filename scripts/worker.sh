#!/bin/bash
# Worker script that runs on each GPU

# Initialize conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate self_play

# Get GPU ID from SLURM
GPU_ID=$SLURM_LOCALID
TASK_ID=$SLURM_PROCID
NODE_NAME=$(hostname)

# Create separate log file for this worker
WORKER_LOG="logs/worker_${SLURM_JOB_ID}_task${TASK_ID}_gpu${GPU_ID}.log"
mkdir -p logs

# Log everything from this worker to separate file
exec > >(tee -a "$WORKER_LOG") 2>&1

echo "==============================================="
echo "Worker $TASK_ID starting at $(date)"
echo "Node: $NODE_NAME"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "Assigned GPU ID: $GPU_ID"
echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo "==============================================="

# Set CUDA device - THIS ENSURES EACH WORKER USES ONLY ITS ASSIGNED GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Verify GPU assignment
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info for this worker:"
    nvidia-smi --id=$GPU_ID --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "GPU $GPU_ID assigned (nvidia-smi query failed, but this is normal)"
fi
echo "==============================================="

# Read tasks from file
TASKS_FILE="tasks_remaining.txt"

# Use a lock file to coordinate task assignment
LOCK_FILE="tasks.lock"

while true; do
    # Acquire lock and get next task
    (
        flock -x 200
        
        # Read first line and remove it from file
        TASK=$(head -n 1 "$TASKS_FILE" 2>/dev/null)
        
        if [ -z "$TASK" ]; then
            # No more tasks
            echo "" > /tmp/worker_${TASK_ID}_task.txt
        else
            # Save task for this worker
            echo "$TASK" > /tmp/worker_${TASK_ID}_task.txt
            
            # Remove the task from the file
            tail -n +2 "$TASKS_FILE" > "${TASKS_FILE}.tmp"
            mv "${TASKS_FILE}.tmp" "$TASKS_FILE"
        fi
    ) 200>"$LOCK_FILE"
    
    # Read the task assigned to this worker
    TASK=$(cat /tmp/worker_${TASK_ID}_task.txt)
    
    if [ -z "$TASK" ]; then
        echo "Worker $TASK_ID (GPU $GPU_ID): No more tasks, exiting at $(date)"
        break
    fi
    
    # Parse task
    IFS='|' read -r MODEL DATASET TOKENS OUTPUT_PATH <<< "$TASK"
    
    echo ""
    echo "=== [$(date)] Worker $TASK_ID (GPU $GPU_ID) ==="
    echo "Processing: $MODEL"
    echo "Dataset: $DATASET"
    echo "Max tokens: $TOKENS"
    echo "Output: $OUTPUT_PATH"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "=============================================="
    
    # Create output directory
    mkdir -p "$OUTPUT_PATH"
    
    # Auto-adjust batch size based on model name
    if [[ "$MODEL" =~ "1.5B"|"2B" ]]; then
        BATCH_SIZE=32
    elif [[ "$MODEL" =~ "7B"|"8B" ]]; then
        BATCH_SIZE=32
    else
        BATCH_SIZE=32
    fi
    
    echo "Running inference (batch_size=$BATCH_SIZE)"
    
    # Run inference
    CUDA_VISIBLE_DEVICES=$GPU_ID python src/run.py \
        --model_path "$MODEL" \
        --dataset "$DATASET" \
        --input_path "data/${DATASET}.jsonl" \
        --output_dir "$OUTPUT_PATH" \
        --max_tokens "$TOKENS" \
        --batch_size "$BATCH_SIZE"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        # Mark task as completed
        touch "${OUTPUT_PATH}/.completed"
        echo "Worker $TASK_ID (GPU $GPU_ID): Task completed successfully"
    else
        echo "Worker $TASK_ID (GPU $GPU_ID): Task failed with exit code $EXIT_CODE"
        # Add task back to the list for retry
        (
            flock -x 200
            echo "$TASK" >> "$TASKS_FILE"
        ) 200>"$LOCK_FILE"
    fi
done

echo ""
echo "==============================================="
echo "Worker $TASK_ID (GPU $GPU_ID) finished at $(date)"
echo "Total tasks completed by this worker"
echo "Log file: $WORKER_LOG"
echo "==============================================="

