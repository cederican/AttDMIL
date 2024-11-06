#!/bin/bash

# Generate a unique session name using the current timestamp
SESSION_NAME="session_$(date +%Y%m%d_%H%M%S)"

# Request a compute node and run the Python script in Apptainer
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer exec --nv pml.sif python \"$1\""

