#!/bin/bash

# Install dependencies for experiments
# This script creates a virtual environment and installs exact working versions

set -e

echo "Setting up experiment environment..."

# Create virtual environment
python3 -m venv .env-experiments
source .env-experiments/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install exact working versions
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate: source .env-experiments/bin/activate"
echo "To run experiments: see EXPERIMENTS.md"
