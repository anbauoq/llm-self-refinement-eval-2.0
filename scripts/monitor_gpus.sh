#!/bin/bash
# Monitor GPU usage across all workers

JOB_ID=$1

if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 <job_id>"
    echo ""
    echo "Example: $0 4122656"
    echo ""
    echo "Your running jobs:"
    squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
    exit 1
fi

echo "Monitoring GPU usage for job $JOB_ID"
echo "=========================================="
echo ""

# Get the node(s) running this job
NODES=$(squeue -j $JOB_ID -h -o "%N")

if [ -z "$NODES" ]; then
    echo "Job $JOB_ID not found or not running"
    exit 1
fi

echo "Job running on node(s): $NODES"
echo ""

# Show GPU usage on the node
echo "GPU Usage on $NODES:"
echo "=========================================="
srun --jobid=$JOB_ID --overlap -w $NODES nvidia-smi

echo ""
echo "=========================================="
echo "Worker Log Files:"
echo "=========================================="
ls -lh logs/worker_${JOB_ID}_*.log 2>/dev/null || echo "No worker logs found yet"

echo ""
echo "=========================================="
echo "Quick Log Summary:"
echo "=========================================="
for log in logs/worker_${JOB_ID}_*.log; do
    if [ -f "$log" ]; then
        echo ""
        echo "=== $(basename $log) ==="
        echo "GPU assignment:"
        grep "Assigned GPU ID:" "$log" | tail -1
        grep "CUDA_VISIBLE_DEVICES set to:" "$log" | tail -1
        echo "Current task:"
        grep "Processing:" "$log" | tail -1
        echo "Status:"
        tail -3 "$log"
    fi
done

