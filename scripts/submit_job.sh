#!/bin/bash
# Convenience script to submit the SLURM job

set -euo pipefail

cd "$(dirname "$0")/.."

# Parse command line arguments
USE_OPTIMIZED="${USE_OPTIMIZED:-true}"
if [ "${1:-}" = "--standard" ] || [ "${1:-}" = "--no-optimize" ]; then
    USE_OPTIMIZED="false"
    shift
elif [ "${1:-}" = "--optimized" ]; then
    USE_OPTIMIZED="true"
    shift
fi

# Export for job submission
export USE_OPTIMIZED

# Initialize conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate self_play

echo "Submitting SLURM job for LLM Self-Refinement Evaluation"
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo "Inference mode: $([ "$USE_OPTIMIZED" = "true" ] && echo "OPTIMIZED" || echo "STANDARD")"
echo ""

# Create logs directory
mkdir -p logs

# Generate initial task list
echo "Generating task list..."
python scripts/generate_tasks.py
echo ""

# Submit the job
JOB_ID=$(sbatch scripts/slurm_run.sh | awk '{print $4}')

echo "Job submitted successfully!"
echo "Job ID: $JOB_ID"
echo ""
echo "Monitor your job with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs with:"
echo "  tail -f logs/slurm_${JOB_ID}.out"
echo ""
echo "Check remaining tasks:"
echo "  wc -l tasks_remaining.txt"
echo ""
echo "Note: Using $([ "$USE_OPTIMIZED" = "true" ] && echo "optimized" || echo "standard") inference"
echo "To change: USE_OPTIMIZED=false ./scripts/submit_job.sh"

