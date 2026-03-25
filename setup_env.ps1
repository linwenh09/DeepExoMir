# DeepExoMir Environment Setup Script (Windows PowerShell)
# Usage: .\setup_env.ps1

Write-Host "=== DeepExoMir Environment Setup ===" -ForegroundColor Cyan

# Step 1: Create conda environment
Write-Host "`n[1/4] Creating conda environment..." -ForegroundColor Yellow
conda create -n deepexomir python=3.11 -y
conda activate deepexomir

# Step 2: Install PyTorch with CUDA support
Write-Host "`n[2/4] Installing PyTorch with CUDA 12.8..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"

# Step 3: Install core dependencies
Write-Host "`n[3/4] Installing core dependencies..." -ForegroundColor Yellow
pip install transformers multimolecule biopython
pip install streamlit pandas numpy scikit-learn
pip install pyarrow pyyaml tensorboard tqdm requests
pip install pyvis networkx plotly openpyxl

# Step 4: Install miRBench (may need special handling)
Write-Host "`n[4/4] Installing miRBench..." -ForegroundColor Yellow
pip install miRBench

# Optional: ViennaRNA (best installed via conda-forge)
Write-Host "`nOptional: Installing ViennaRNA via conda-forge..." -ForegroundColor Yellow
conda install -c conda-forge viennarna -y

# Install DeepExoMir as editable package
Write-Host "`nInstalling DeepExoMir in development mode..." -ForegroundColor Yellow
pip install -e ".[dev]"

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "Activate with: conda activate deepexomir"
Write-Host "Run tests with: pytest tests/"
Write-Host "Start web UI with: streamlit run deepexomir/webapp/app.py"
