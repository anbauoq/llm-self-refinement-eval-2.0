#!/bin/bash
# Cancel the entire job chain

cd "$(dirname "$0")/.."

echo "========================================"
echo "Cancel Job Chain"
echo "========================================"
echo ""

# Check for active jobs
ACTIVE_JOBS=$(squeue -u $USER -h -o "%i")

if [ -z "$ACTIVE_JOBS" ]; then
    echo "No active jobs found for user $USER"
    exit 0
fi

echo "Active jobs:"
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
echo ""

read -p "Cancel ALL your jobs? (yes/no): " -r
echo

if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Cancelling all jobs..."
    scancel -u $USER
    
    # Remove chain file to prevent auto-resubmission
    if [ -f "logs/job_chain.txt" ]; then
        echo "Archiving chain file..."
        mv logs/job_chain.txt "logs/job_chain_cancelled_$(date '+%Y%m%d_%H%M%S').txt"
    fi
    
    echo ""
    echo "✓ All jobs cancelled"
    echo "✓ Chain file archived"
    echo ""
    echo "To resume later, run: ./scripts/submit_job.sh"
else
    echo "Cancelled. No jobs were stopped."
fi

