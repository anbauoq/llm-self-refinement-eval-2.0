#!/bin/bash
# Monitor the job chain progress

cd "$(dirname "$0")/.."

CHAIN_FILE="logs/job_chain.txt"

echo "========================================"
echo "Job Chain Status Monitor"
echo "========================================"
echo ""

# Check if chain is active
if [ ! -f "$CHAIN_FILE" ]; then
    echo "No active job chain found."
    echo ""
    echo "Check completed chains:"
    ls -lt logs/job_chain_completed_*.txt 2>/dev/null | head -5
    exit 0
fi

# Display chain information
echo "Active Job Chain:"
echo "----------------"
cat "$CHAIN_FILE" | while IFS='|' read -r chain_num job_id timestamp; do
    echo "  Chain #$chain_num: Job $job_id (started $timestamp)"
done
echo ""

# Get current chain number
CURRENT_CHAIN=$(tail -1 "$CHAIN_FILE" | cut -d'|' -f1)
LATEST_JOB=$(tail -1 "$CHAIN_FILE" | cut -d'|' -f2)

echo "Current Status:"
echo "---------------"
echo "  Chain length: $CURRENT_CHAIN job(s)"
echo "  Latest job ID: $LATEST_JOB"
echo ""

# Check if latest job is still running
RUNNING_JOBS=$(squeue -u $USER -h -o "%i" | grep "$LATEST_JOB")
if [ -n "$RUNNING_JOBS" ]; then
    echo "  Status: ✓ RUNNING"
    squeue -j $LATEST_JOB -o "  %.18i %.9P %.30j %.8T %.10M %.6D %R"
else
    echo "  Status: ⏸ NOT RUNNING (completed or pending next submission)"
fi

echo ""

# Show progress
if [ -f "tasks_remaining.txt" ] && [ -f "tasks.txt" ]; then
    TOTAL=$(wc -l < tasks.txt)
    REMAINING=$(wc -l < tasks_remaining.txt)
    COMPLETED=$((TOTAL - REMAINING))
    PERCENT=$((COMPLETED * 100 / TOTAL))
    
    echo "Task Progress:"
    echo "--------------"
    echo "  Total: $TOTAL tasks"
    echo "  Completed: $COMPLETED"
    echo "  Remaining: $REMAINING"
    echo "  Progress: $PERCENT%"
    
    # Progress bar
    BAR_WIDTH=50
    FILLED=$((PERCENT * BAR_WIDTH / 100))
    EMPTY=$((BAR_WIDTH - FILLED))
    printf "  ["
    printf "%${FILLED}s" | tr ' ' '='
    printf "%${EMPTY}s" | tr ' ' '-'
    printf "] $PERCENT%%\n"
fi

echo ""
echo "========================================"
echo "Quick Commands:"
echo "========================================"
echo "  Watch queue:     watch -n 5 squeue -u \$USER"
echo "  Latest log:      tail -f logs/slurm_${LATEST_JOB}.out"
echo "  Worker logs:     ls -lh logs/worker_${LATEST_JOB}_*.log"
echo "  GPU monitor:     ./scripts/monitor_gpus.sh $LATEST_JOB"
echo "  Cancel chain:    scancel -u \$USER"
echo ""

