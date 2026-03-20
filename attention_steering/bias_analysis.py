"""
Bias analysis through symmetric-antisymmetric attention decomposition.

Uses CrowS-Pairs dataset: minimal sentence pairs that differ only by a
demographic group, labeled as stereotypical vs. anti-stereotypical.

We compare the spectral signatures of attention between paired sentences
to find correlations between eigenvalue structure and bias.
"""

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm

from .decompose import AttentionDecomposer
from .extract import AttentionExtractor, AttentionMaps


BIAS_CATEGORIES = {
    "race-color": "Race / Color",
    "gender": "Gender",
    "socioeconomic": "Socioeconomic",
    "nationality": "Nationality",
    "religion": "Religion",
    "age": "Age",
    "sexual-orientation": "Sexual Orientation",
    "physical-appearance": "Physical Appearance",
    "disability": "Disability",
}


@dataclass
class PairSpectralResult:
    """Spectral analysis result for a single stereotypical/anti-stereotypical pair."""

    pair_idx: int
    bias_type: str

    # Asymmetry scores (mean over layers & heads)
    stereo_asymmetry: float
    antistereo_asymmetry: float
    asymmetry_delta: float  # stereo - antistereo

    # Symmetric eigenvalue stats
    stereo_sym_top_eigenvalue: float
    antistereo_sym_top_eigenvalue: float
    stereo_sym_spectral_entropy: float
    antistereo_sym_spectral_entropy: float

    # Antisymmetric eigenvalue stats
    stereo_asym_top_eigenvalue: float
    antistereo_asym_top_eigenvalue: float
    stereo_asym_spectral_entropy: float
    antistereo_asym_spectral_entropy: float

    # Per-layer asymmetry scores
    stereo_layer_asymmetry: List[float] = field(default_factory=list)
    antistereo_layer_asymmetry: List[float] = field(default_factory=list)


def load_crows_pairs(
    categories: Optional[List[str]] = None,
    max_pairs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load CrowS-Pairs bias benchmark dataset.

    Args:
        categories: Filter to specific bias types (e.g., ["gender", "race-color"]).
                    None = all categories.
        max_pairs: Limit number of pairs (useful for quick experiments).

    Returns:
        DataFrame with columns: sent_more (stereotypical), sent_less (anti-stereotypical),
        bias_type, stereo_antistereo (direction label).
    """
    from datasets import load_dataset

    dataset = load_dataset("nyu-mll/crows_pairs", split="test")
    df = pd.DataFrame(dataset)

    if categories:
        df = df[df["bias_type"].isin(categories)].reset_index(drop=True)

    if max_pairs:
        df = df.head(max_pairs).reset_index(drop=True)

    return df


def _spectral_entropy(eigenvalues: torch.Tensor) -> float:
    """
    Shannon entropy of the eigenvalue magnitude distribution.
    High entropy = eigenvalues spread evenly (diffuse attention).
    Low entropy = few dominant eigenvalues (concentrated attention).
    """
    magnitudes = eigenvalues.abs().float()
    total = magnitudes.sum() + 1e-10
    probs = magnitudes / total
    probs = probs[probs > 1e-10]
    entropy = -(probs * probs.log()).sum().item()
    return entropy


def analyze_pair(
    extractor: AttentionExtractor,
    stereo_text: str,
    antistereo_text: str,
    pair_idx: int,
    bias_type: str,
    layers_to_analyze: Optional[List[int]] = None,
) -> PairSpectralResult:
    """
    Run full spectral decomposition comparison on a single sentence pair.
    """
    decomposer = AttentionDecomposer()

    attn_stereo = extractor.extract(stereo_text)
    attn_antistereo = extractor.extract(antistereo_text)

    num_layers = attn_stereo.num_layers
    num_heads = attn_stereo.num_heads

    if layers_to_analyze is None:
        layers_to_analyze = list(range(num_layers))

    stereo_asymmetries = []
    antistereo_asymmetries = []
    stereo_sym_eigs_all = []
    antistereo_sym_eigs_all = []
    stereo_asym_eigs_all = []
    antistereo_asym_eigs_all = []
    stereo_layer_asym = []
    antistereo_layer_asym = []

    for layer in layers_to_analyze:
        layer_stereo_asym = []
        layer_antistereo_asym = []

        for head in range(num_heads):
            A_s = attn_stereo.get(layer, head)
            A_a = attn_antistereo.get(layer, head)

            asym_s = decomposer.asymmetry_score(A_s).item()
            asym_a = decomposer.asymmetry_score(A_a).item()
            stereo_asymmetries.append(asym_s)
            antistereo_asymmetries.append(asym_a)
            layer_stereo_asym.append(asym_s)
            layer_antistereo_asym.append(asym_a)

            S_s, K_s = decomposer.decompose(A_s)
            S_a, K_a = decomposer.decompose(A_a)

            stereo_sym_eigs_all.append(decomposer.eigenspectrum_symmetric(S_s))
            antistereo_sym_eigs_all.append(decomposer.eigenspectrum_symmetric(S_a))
            stereo_asym_eigs_all.append(decomposer.eigenspectrum_antisymmetric(K_s))
            antistereo_asym_eigs_all.append(decomposer.eigenspectrum_antisymmetric(K_a))

        stereo_layer_asym.append(np.mean(layer_stereo_asym))
        antistereo_layer_asym.append(np.mean(layer_antistereo_asym))

    mean_stereo_asym = np.mean(stereo_asymmetries)
    mean_antistereo_asym = np.mean(antistereo_asymmetries)

    stereo_sym_top = np.mean([e[0].item() for e in stereo_sym_eigs_all])
    antistereo_sym_top = np.mean([e[0].item() for e in antistereo_sym_eigs_all])
    stereo_asym_top = np.mean([e[0].abs().item() for e in stereo_asym_eigs_all])
    antistereo_asym_top = np.mean([e[0].abs().item() for e in antistereo_asym_eigs_all])

    stereo_sym_entropy = np.mean([_spectral_entropy(e) for e in stereo_sym_eigs_all])
    antistereo_sym_entropy = np.mean([_spectral_entropy(e) for e in antistereo_sym_eigs_all])
    stereo_asym_entropy = np.mean([_spectral_entropy(e) for e in stereo_asym_eigs_all])
    antistereo_asym_entropy = np.mean([_spectral_entropy(e) for e in antistereo_asym_eigs_all])

    return PairSpectralResult(
        pair_idx=pair_idx,
        bias_type=bias_type,
        stereo_asymmetry=mean_stereo_asym,
        antistereo_asymmetry=mean_antistereo_asym,
        asymmetry_delta=mean_stereo_asym - mean_antistereo_asym,
        stereo_sym_top_eigenvalue=stereo_sym_top,
        antistereo_sym_top_eigenvalue=antistereo_sym_top,
        stereo_sym_spectral_entropy=stereo_sym_entropy,
        antistereo_sym_spectral_entropy=antistereo_sym_entropy,
        stereo_asym_top_eigenvalue=stereo_asym_top,
        antistereo_asym_top_eigenvalue=antistereo_asym_top,
        stereo_asym_spectral_entropy=stereo_asym_entropy,
        antistereo_asym_spectral_entropy=antistereo_asym_entropy,
        stereo_layer_asymmetry=stereo_layer_asym,
        antistereo_layer_asymmetry=antistereo_layer_asym,
    )


def run_bias_analysis(
    extractor: AttentionExtractor,
    categories: Optional[List[str]] = None,
    max_pairs: Optional[int] = None,
    layers_to_analyze: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, List[PairSpectralResult]]:
    """
    Run spectral bias analysis on CrowS-Pairs dataset.

    Returns:
        (results_df, raw_results) — DataFrame of aggregated metrics and list of per-pair results.
    """
    df = load_crows_pairs(categories=categories, max_pairs=max_pairs)
    results: List[PairSpectralResult] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing pairs"):
        try:
            result = analyze_pair(
                extractor=extractor,
                stereo_text=row["sent_more"],
                antistereo_text=row["sent_less"],
                pair_idx=idx,
                bias_type=row["bias_type"],
                layers_to_analyze=layers_to_analyze,
            )
            results.append(result)
        except Exception as e:
            print(f"Skipping pair {idx}: {e}")
            continue

    records = []
    for r in results:
        records.append({
            "pair_idx": r.pair_idx,
            "bias_type": r.bias_type,
            "stereo_asymmetry": r.stereo_asymmetry,
            "antistereo_asymmetry": r.antistereo_asymmetry,
            "asymmetry_delta": r.asymmetry_delta,
            "stereo_sym_top_eig": r.stereo_sym_top_eigenvalue,
            "antistereo_sym_top_eig": r.antistereo_sym_top_eigenvalue,
            "sym_top_eig_delta": r.stereo_sym_top_eigenvalue - r.antistereo_sym_top_eigenvalue,
            "stereo_sym_entropy": r.stereo_sym_spectral_entropy,
            "antistereo_sym_entropy": r.antistereo_sym_spectral_entropy,
            "sym_entropy_delta": r.stereo_sym_spectral_entropy - r.antistereo_sym_spectral_entropy,
            "stereo_asym_top_eig": r.stereo_asym_top_eigenvalue,
            "antistereo_asym_top_eig": r.antistereo_asym_top_eigenvalue,
            "asym_top_eig_delta": r.stereo_asym_top_eigenvalue - r.antistereo_asym_top_eigenvalue,
            "stereo_asym_entropy": r.stereo_asym_spectral_entropy,
            "antistereo_asym_entropy": r.antistereo_asym_spectral_entropy,
            "asym_entropy_delta": r.stereo_asym_spectral_entropy - r.antistereo_asym_spectral_entropy,
        })

    results_df = pd.DataFrame(records)
    return results_df, results


def summarize_by_category(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate spectral bias metrics by bias category."""
    delta_cols = [c for c in results_df.columns if "delta" in c]
    summary = results_df.groupby("bias_type")[delta_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    return summary.sort_values("asymmetry_delta_mean", ascending=False)
