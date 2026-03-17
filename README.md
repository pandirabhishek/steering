# Attention Steering via Symmetric-Antisymmetric Decomposition

Steering LLM/VLLM behavior by decomposing attention maps into symmetric and antisymmetric components, then selectively modifying their eigenvalue spectra.

## Core Idea

Any attention matrix **A** ∈ ℝⁿˣⁿ can be uniquely decomposed as:

$$A = S + K$$

where:
- **S = (A + Aᵀ) / 2** — symmetric component with **real eigenvalues**
- **K = (A - Aᵀ) / 2** — antisymmetric (skew-symmetric) component with **purely imaginary eigenvalues**

### Interpretation

| Component | Eigenvalues | Captures |
|-----------|-------------|----------|
| **S** (symmetric) | Real | Mutual/bidirectional attention — tokens that attend to each other equally |
| **K** (antisymmetric) | Imaginary | Directional asymmetry — information flow direction between tokens |

### Steering Strategies

- **Eigenvalue scaling** — amplify or suppress specific modes of S or K
- **Spectral filtering** — keep only top-k eigenvalues for low-rank steering
- **Phase modulation** — rotate the imaginary eigenvalues of K to redirect information flow
- **Component reweighting** — adjust the relative contribution of S vs K

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
steering/
├── attention_steering/
│   ├── __init__.py
│   ├── decompose.py      # Symmetric/antisymmetric decomposition & spectral ops
│   ├── extract.py         # Extract attention maps from HuggingFace models
│   ├── steer.py           # Steering hooks for inference-time intervention
│   └── viz.py             # Visualization utilities
├── notebooks/
│   └── 01_attention_decomposition.ipynb
├── requirements.txt
└── README.md
```

## Quick Start

```python
from attention_steering import AttentionDecomposer, AttentionExtractor, AttentionSteerer

# Extract attention from a model
extractor = AttentionExtractor("gpt2")
attentions = extractor.extract("The cat sat on the mat")

# Decompose into symmetric + antisymmetric
decomposer = AttentionDecomposer()
S, K = decomposer.decompose(attentions[0][0])  # layer 0, head 0

# Analyze eigenvalue spectra
real_eigenvalues = decomposer.eigenspectrum_symmetric(S)
imag_eigenvalues = decomposer.eigenspectrum_antisymmetric(K)

# Steer by modifying components
steerer = AttentionSteerer(extractor.model, extractor.tokenizer)
output = steerer.generate_with_steering(
    "The cat sat on",
    layer=6,
    symmetric_scale=1.5,    # boost mutual attention
    antisymmetric_scale=0.5  # dampen directional asymmetry
)
```

## License

MIT
