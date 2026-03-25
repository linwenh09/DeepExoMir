# DeepExoMir

A hybrid deep learning framework for microRNA target prediction, integrating RNA language model embeddings with duplex graph attention.

## Key Features

- **RNA Language Model Integration**: Frozen RiNALMo-giga (650M params) embeddings with PCA dimensionality reduction
- **Hybrid Encoder**: 8-layer architecture alternating BiConvGate and cross-attention layers
- **DuplexGAT**: Graph attention network modeling the miRNA-target duplex as a nucleotide-level graph
- **Interaction Pooling**: Multi-head attention mechanism capturing pairwise sequence interactions
- **33 Biological Features**: Thermodynamic stability, evolutionary conservation (PhyloP), RNA secondary structure

## Performance

Evaluated on the [miRBench](https://github.com/katarinagresova/miRBench) benchmark with experimentally validated CLIP-seq negatives:

| Method | Hejret | Klimentova | Manakov | Mean AU-PRC |
|--------|:------:|:----------:|:-------:|:-----------:|
| Best retrained CNN | 0.84 | 0.82 | 0.81 | 0.82 |
| **DeepExoMir** | **0.85** | **0.87** | **0.85** | **0.86** |

## Installation

```bash
# Clone the repository
git clone https://github.com/linwenh09/DeepExoMir.git
cd DeepExoMir

# Create conda environment
conda create -n deepexomir python=3.11
conda activate deepexomir

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Prediction
```python
from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8
import yaml, torch

# Load model
with open("configs/model_config_v19.yaml") as f:
    config = yaml.safe_load(f)
model = DeepExoMirModelV8(config)
ckpt = torch.load("checkpoints/v19/checkpoint_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
```

### Training
```bash
python scripts/train.py \
    --config configs/train_config_v19.yaml \
    --model-config configs/model_config_v19.yaml \
    --embeddings-dir data/embeddings_cache_pca256
```

### Evaluation on miRBench
```bash
python scripts/evaluate_mirbench.py \
    --checkpoint checkpoints/v19/checkpoint_best.pt \
    --model-config configs/model_config_v19.yaml \
    --use-backbone
```

## Model Architecture

```
miRNA (22 nt) + Target (50 nt)
        |
  RiNALMo-giga (frozen) -> PCA 1280->256
        |
  8-Layer Hybrid Encoder (4 BiConvGate + 4 CrossAttention)
        |
  +-- Interaction Pooling (512d)
  +-- Backbone Feature MLP (256d)
  +-- 6-ch Base-Pairing CNN (128d)
  +-- Contact Map CNN (128d)
  +-- DuplexGAT (128d)
  +-- Structural MLP (64d)
        |
  Concatenate (1216d)
        |
  MoE Classifier (4 experts, top-2)
        |
  P(target)
```

## Project Structure

```
DeepExoMir/
├── deepexomir/           # Core model code
│   ├── model/            # Model architecture
│   │   ├── deepexomir_v8.py    # Main model
│   │   ├── hybrid_encoder.py   # BiConvGate + CrossAttention
│   │   ├── duplex_gat.py       # Graph attention network
│   │   ├── interaction_pooling.py
│   │   ├── moe_classifier.py
│   │   └── ...
│   └── data/             # Dataset loading
├── configs/              # Model and training configs
├── scripts/              # Training, evaluation, analysis
├── manuscript/           # Paper figures
└── app/                  # Streamlit web interface
```

## Citation

If you use DeepExoMir in your research, please cite:

```bibtex
@article{lin2026deepexomir,
  title={DeepExoMir: A Hybrid Deep Learning Framework Integrating RNA Language Model Embeddings and Duplex Graph Attention for MicroRNA Target Prediction},
  author={Lin, Wen-Hsien},
  journal={Bioinformatics},
  year={2026}
}
```

## License

MIT License

## Contact

Wen-Hsien Lin - BryceLin@bionetTX.com

AI and Data Applications Division, GGA Corp.
