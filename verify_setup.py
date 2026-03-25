"""Verify DeepExoMir environment setup."""
import sys

print(f"Python {sys.version}")
errors = []

# Core ML
try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | {gpu}")
except Exception as e:
    errors.append(f"torch: {e}")

try:
    import transformers
    print(f"transformers {transformers.__version__}")
except Exception as e:
    errors.append(f"transformers: {e}")

try:
    import multimolecule
    print(f"multimolecule OK")
except Exception as e:
    # multimolecule 0.0.9 incompatible with transformers 5.x
    # Not needed at runtime - we use trust_remote_code=True
    print(f"multimolecule SKIP (not needed, trust_remote_code=True handles it)")

# Bio
try:
    import Bio
    print(f"biopython {Bio.__version__}")
except Exception as e:
    errors.append(f"biopython: {e}")

try:
    import RNA
    print(f"ViennaRNA OK")
except Exception as e:
    errors.append(f"ViennaRNA: {e}")

# Data
try:
    import pandas, numpy, sklearn, pyarrow
    print(f"pandas {pandas.__version__} | numpy {numpy.__version__} | sklearn {sklearn.__version__}")
except Exception as e:
    errors.append(f"data libs: {e}")

# Web UI
try:
    import streamlit
    print(f"streamlit {streamlit.__version__}")
except Exception as e:
    errors.append(f"streamlit: {e}")

# Visualization
try:
    import networkx, pyvis, plotly
    print(f"networkx {networkx.__version__} | pyvis OK | plotly {plotly.__version__}")
except Exception as e:
    errors.append(f"viz libs: {e}")

# miRBench
try:
    import miRBench
    print(f"miRBench OK")
except Exception as e:
    errors.append(f"miRBench: {e}")

# DeepExoMir package
try:
    import deepexomir
    from deepexomir.model import DeepExoMirModel
    from deepexomir.data.dataset import MiRNATargetDataset
    from deepexomir.annotation.aesthetic_scorer import AestheticScorer
    print(f"deepexomir OK (all submodules)")
except Exception as e:
    errors.append(f"deepexomir: {e}")

print()
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")
else:
    print("=== All packages verified successfully! ===")
