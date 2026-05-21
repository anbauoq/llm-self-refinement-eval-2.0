#!/bin/bash
#SBATCH -A llmservice_nemo_reasoning
#SBATCH -p batch
#SBATCH -J llmservice_nemo_reasoning-self_refine:eval_all_models
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err

# Signal handler for graceful timeout (triggered 5 min before time limit)
handle_timeout() {
    echo ""
    echo "=========================================="
    echo "⏰ TIME LIMIT APPROACHING - Triggering graceful resubmission"
    echo "=========================================="
    echo "Signal received at: $(date)"
    echo "Job #$CHAIN_NUM in chain"
    
    # Change to project directory
    cd /home/earakelyan/llm-self-refinement-eval-2.0
    
    # Kill worker processes gracefully
    echo "Stopping workers..."
    if [ ! -z "$SRUN_PID" ]; then
        kill $SRUN_PID 2>/dev/null
    fi
    sleep 5
    
    # Resubmit if tasks remain
    REMAINING=$(wc -l < tasks_remaining.txt 2>/dev/null || echo 0)
    if [ "$REMAINING" -gt 0 ]; then
        echo "Tasks remaining: $REMAINING"
        echo "Auto-resubmitting..."
        
        SUBMIT_OUTPUT=$(sbatch scripts/slurm_run.sh 2>&1)
        if echo "$SUBMIT_OUTPUT" | grep -q "Submitted batch job"; then
            NEW_JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
            echo "✅ Resubmitted as job $NEW_JOB_ID"
            
            # Update chain file
            CHAIN_FILE="logs/job_chain.txt"
            if [ -f "$CHAIN_FILE" ]; then
                LAST_CHAIN_NUM=$(tail -1 "$CHAIN_FILE" | cut -d'|' -f1)
                echo "$((LAST_CHAIN_NUM+1))|$NEW_JOB_ID|$(date '+%Y-%m-%d %H:%M:%S')" >> "$CHAIN_FILE"
            fi
        else
            echo "❌ Resubmission failed: $SUBMIT_OUTPUT"
        fi
    else
        echo "✅ All tasks completed!"
    fi
    
    exit 0
}

# Set up signal trap
trap 'handle_timeout' USR1

# Initialize conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate self_play

# Set cache directories to Lustre filesystem
export PIP_CACHE_DIR=/lustre/fsw/llmservice_nemo_reasoning/earakelyan/.cache/pip
export HF_HOME=/lustre/fsw/llmservice_nemo_reasoning/earakelyan/.cache/huggingface
export TRANSFORMERS_CACHE=/lustre/fsw/llmservice_nemo_reasoning/earakelyan/.cache/huggingface

echo "Job started at $(date)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Cache directories:"
echo "  HF_HOME: $HF_HOME"
echo ""

# Change to project directory
cd /home/earakelyan/llm-self-refinement-eval-2.0

# Create logs directory if it doesn't exist
mkdir -p logs

# Job chain tracking
CHAIN_FILE="logs/job_chain.txt"
MAX_CHAIN_LENGTH=20  # Safety limit: max 20 resubmissions

# Initialize or update chain file
if [ ! -f "$CHAIN_FILE" ]; then
    echo "1|$SLURM_JOB_ID|$(date '+%Y-%m-%d %H:%M:%S')" > "$CHAIN_FILE"
    CHAIN_NUM=1
else
    CHAIN_NUM=$(tail -1 "$CHAIN_FILE" | cut -d'|' -f1)
    CHAIN_NUM=$((CHAIN_NUM + 1))
    echo "$CHAIN_NUM|$SLURM_JOB_ID|$(date '+%Y-%m-%d %H:%M:%S')" >> "$CHAIN_FILE"
fi

echo "=========================================="
echo "Job Chain #$CHAIN_NUM (Job ID: $SLURM_JOB_ID)"
echo "=========================================="

# Check chain length safety limit
if [ "$CHAIN_NUM" -gt "$MAX_CHAIN_LENGTH" ]; then
    echo "ERROR: Maximum chain length ($MAX_CHAIN_LENGTH) exceeded!"
    echo "This job chain has been running too long. Please investigate."
    echo "Check logs/job_chain.txt for history"
    exit 1
fi

# Generate task list (filters out completed tasks)
echo "Generating task list..."
python scripts/generate_tasks.py

# Check if there are tasks remaining
REMAINING=$(wc -l < tasks_remaining.txt)
TOTAL=$(wc -l < tasks.txt)
COMPLETED=$((TOTAL - REMAINING))

echo ""
echo "Task Status:"
echo "  Total tasks: $TOTAL"
echo "  Completed: $COMPLETED"
echo "  Remaining: $REMAINING"
echo "  Progress: $((COMPLETED * 100 / TOTAL))%"
echo ""

if [ "$REMAINING" -eq 0 ]; then
    echo "=========================================="
    echo "SUCCESS: All experiments completed!"
    echo "Job chain finished after $CHAIN_NUM job(s)"
    echo "=========================================="
    
    # Archive the chain file
    mv "$CHAIN_FILE" "logs/job_chain_completed_$(date '+%Y%m%d_%H%M%S').txt"
    exit 0
fi

echo "Starting work on $REMAINING remaining tasks..."

# Launch workers in parallel using SLURM's srun
echo "Launching $SLURM_NTASKS workers..."
srun --ntasks=$SLURM_NTASKS --nodes=1 bash scripts/worker.sh &
SRUN_PID=$!

# Wait for workers to complete (or timeout signal)
wait $SRUN_PID

# Check if any tasks remain after job completion
echo ""
echo "=========================================="
echo "Job #$CHAIN_NUM finished at $(date)"
echo "=========================================="

python scripts/generate_tasks.py
REMAINING_AFTER=$(wc -l < tasks_remaining.txt)
TOTAL_AFTER=$(wc -l < tasks.txt)
COMPLETED_AFTER=$((TOTAL_AFTER - REMAINING_AFTER))
COMPLETED_THIS_JOB=$((REMAINING - REMAINING_AFTER))

echo ""
echo "Job #$CHAIN_NUM Summary:"
echo "  Tasks completed in this job: $COMPLETED_THIS_JOB"
echo "  Total completed: $COMPLETED_AFTER / $TOTAL_AFTER"
echo "  Remaining: $REMAINING_AFTER"
echo "  Overall progress: $((COMPLETED_AFTER * 100 / TOTAL_AFTER))%"
echo ""

# Auto-resubmit if tasks remain
if [ "$REMAINING_AFTER" -gt 0 ]; then
    echo "=========================================="
    echo "Tasks still remaining. Auto-resubmitting..."
    echo "=========================================="
    
    # Small delay to avoid rapid resubmission
    sleep 5
    
    # Resubmit the job
    SUBMIT_OUTPUT=$(sbatch scripts/slurm_run.sh 2>&1)
    if [ $? -eq 0 ]; then
        NEW_JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $4}')
        echo "✓ Successfully resubmitted as job ID: $NEW_JOB_ID"
        echo "  Next job will be chain #$((CHAIN_NUM + 1))"
        echo "  Monitor with: squeue -j $NEW_JOB_ID"
        echo "  View logs: tail -f logs/slurm_${NEW_JOB_ID}.out"
    else
        echo "✗ ERROR: Failed to resubmit job"
        echo "  Error: $SUBMIT_OUTPUT"
        echo "  You may need to manually resubmit: sbatch scripts/slurm_run.sh"
        exit 1
    fi
else
    echo "=========================================="
    echo "🎉 SUCCESS: All $TOTAL_AFTER tasks completed!"
    echo "=========================================="
    echo ""
    echo "Job chain completed after $CHAIN_NUM job(s)"
    echo "Start time: $(head -1 $CHAIN_FILE | cut -d'|' -f3)"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "Chain history saved to: logs/job_chain_completed_$(date '+%Y%m%d_%H%M%S').txt"
    
    # Archive the chain file
    mv "$CHAIN_FILE" "logs/job_chain_completed_$(date '+%Y%m%d_%H%M%S').txt"
    
    echo ""
    echo "Next steps:"
    echo "  1. Analyze results: python src/analysis.py --results_root results/all_results"
    echo "  2. View summary: cat results/all_results/summary.txt"
fi

