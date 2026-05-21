#!/bin/bash
# Setup the self_play conda environment with all required packages

set -euo pipefail

echo "Setting up self_play conda environment..."

# Initialize conda
source ~/miniforge3/etc/profile.d/conda.sh

# Create/update the self_play environment with Python and packages
conda install -n self_play python=3.10 -y

# Activate the environment
conda activate self_play

# Install required packages
pip install -r requirements.txt

echo ""
echo "Environment setup complete!"
echo ""
echo "To activate: conda activate self_play"
echo "Test with: which python && python --version"

