#!/bin/bash
# Helper script for training on Lambda instances with virtual display

set -e

# Check if Xvfb is installed
if ! command -v Xvfb &> /dev/null; then
    echo "Xvfb not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y xvfb
fi

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "qwop" ]]; then
    echo "Activating conda environment 'qwop'..."
    eval "$(conda shell.bash hook)"
    conda activate qwop
fi

# Default to PPO if no argument provided
COMMAND=${1:-train_ppo}

echo "=========================================="
echo "Running: qwop-gym $COMMAND"
echo "With virtual display (Xvfb)"
echo "=========================================="
echo ""

# Run with Xvfb
xvfb-run -a qwop-gym "$COMMAND" "${@:2}"
