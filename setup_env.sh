#!/bin/bash
# DeepExoMir Environment Setup Script (Linux/Mac/WSL)
# Usage: bash setup_env.sh

echo "=== DeepExoMir Environment Setup ==="

# Step 1: Create conda environment
echo -e "\n[1/4] Creating conda environment..."
conda create -n deepexomir python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate deepexomir

# Step 2: Install PyTorch with CUDA support
echo -e "\n[2/4] Installing PyTorch with CUDA 12.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Step 3: Install core dependencies
echo -e "\n[3/4] Installing core dependencies..."
pip install transformers multimolecule biopython
pip install streamlit pandas numpy scikit-learn
pip install pyarrow pyyaml tensorboard tqdm requests
pip install pyvis networkx plotly openpyxl

# Step 4: Install miRBench
echo -e "\n[4/4] Installing miRBench..."
pip install miRBench

# Optional: ViennaRNA
echo -e "\nInstalling ViennaRNA..."
conda install -c conda-forge viennarna -y

# Install editable
pip install -e ".[dev]"

echo -e "\n=== Setup Complete ==="
echo "Activate: conda activate deepexomir"
