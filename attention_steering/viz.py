"""
Visualization utilities for attention decomposition analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Optional, List, Tuple, Union
from .decompose import AttentionDecomposer


def plot_attention_decomposition(
    attention: Union[torch.Tensor, np.ndarray],
    tokens: Optional[List[str]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (18, 5),
    cmap_sym: str = "RdBu_r",
    cmap_asym: str = "PiYG",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot original attention, symmetric component, and antisymmetric component side by side.
    """
    decomposer = AttentionDecomposer()
    A = decomposer._to_tensor(attention).float().cpu()
    S, K = decomposer.decompose(A)

    A_np = A.numpy()
    S_np = S.numpy()
    K_np = K.numpy()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    _plot_matrix(axes[0], A_np, tokens, "Original Attention (A)", "viridis")
    _plot_matrix(axes[1], S_np, tokens, "Symmetric (S) — Real eigenvalues", cmap_sym)
    _plot_matrix(axes[2], K_np, tokens, "Antisymmetric (K) — Imaginary eigenvalues", cmap_asym)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_eigenvalue_spectra(
    attention: Union[torch.Tensor, np.ndarray],
    title: str = "Eigenvalue Spectra",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot eigenvalue spectra of symmetric and antisymmetric components.
    """
    decomposer = AttentionDecomposer()
    S, K = decomposer.decompose(attention)

    real_eigs = decomposer.eigenspectrum_symmetric(S).cpu().numpy()
    imag_eigs = decomposer.eigenspectrum_antisymmetric(K).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.bar(range(len(real_eigs)), real_eigs, color="#2196F3", alpha=0.8, edgecolor="navy")
    ax1.set_title("Symmetric Eigenvalues (Real)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Eigenvalue (λ)")
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.grid(axis="y", alpha=0.3)

    colors = ["#E91E63" if v >= 0 else "#4CAF50" for v in imag_eigs]
    ax2.bar(range(len(imag_eigs)), imag_eigs, color=colors, alpha=0.8, edgecolor="darkred")
    ax2.set_title("Antisymmetric Eigenvalues (Imaginary part)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Im(λ)")
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_asymmetry_across_layers(
    attentions: torch.Tensor,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot asymmetry score (ratio of antisymmetric energy) across all layers and heads.

    Args:
        attentions: Tensor of shape (num_layers, num_heads, seq_len, seq_len).
    """
    decomposer = AttentionDecomposer()
    num_layers, num_heads = attentions.shape[:2]

    scores = torch.zeros(num_layers, num_heads)
    for layer in range(num_layers):
        for head in range(num_heads):
            scores[layer, head] = decomposer.asymmetry_score(
                attentions[layer, head]
            )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        scores.numpy(),
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        xticklabels=[f"H{i}" for i in range(num_heads)],
        yticklabels=[f"L{i}" for i in range(num_layers)],
        cbar_kws={"label": "Asymmetry Score"},
    )
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        "Attention Asymmetry Score (0=symmetric, 1=antisymmetric)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_steering_comparison(
    original: Union[torch.Tensor, np.ndarray],
    steered: Union[torch.Tensor, np.ndarray],
    tokens: Optional[List[str]] = None,
    title: str = "Steering Effect",
    figsize: Tuple[int, int] = (18, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare original vs steered attention and show the difference.
    """
    decomposer = AttentionDecomposer()
    orig_np = decomposer._to_tensor(original).float().cpu().numpy()
    steer_np = decomposer._to_tensor(steered).float().cpu().numpy()
    diff_np = steer_np - orig_np

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    _plot_matrix(axes[0], orig_np, tokens, "Original", "viridis")
    _plot_matrix(axes[1], steer_np, tokens, "Steered", "viridis")
    _plot_matrix(axes[2], diff_np, tokens, "Difference (Steered - Original)", "RdBu_r")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_eigenvalue_evolution(
    attentions: torch.Tensor,
    component: str = "symmetric",
    top_k: int = 5,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Track how top eigenvalues evolve across layers for a specific head.

    Args:
        attentions: Shape (num_layers, seq_len, seq_len) — single head across layers.
        component: "symmetric" or "antisymmetric".
        top_k: Number of top eigenvalues to track.
    """
    decomposer = AttentionDecomposer()
    num_layers = attentions.shape[0]

    eigenvalue_tracks = []
    for layer in range(num_layers):
        S, K = decomposer.decompose(attentions[layer])
        if component == "symmetric":
            eigs = decomposer.eigenspectrum_symmetric(S).cpu().numpy()
        else:
            eigs = decomposer.eigenspectrum_antisymmetric(K).cpu().numpy()
        eigenvalue_tracks.append(eigs[:top_k])

    eigenvalue_tracks = np.array(eigenvalue_tracks)

    fig, ax = plt.subplots(figsize=figsize)
    for k in range(top_k):
        ax.plot(
            range(num_layers),
            eigenvalue_tracks[:, k],
            marker="o",
            label=f"λ_{k+1}",
            linewidth=2,
            markersize=5,
        )

    comp_label = "Real" if component == "symmetric" else "Imaginary"
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(f"Eigenvalue ({comp_label})", fontsize=12)
    ax.set_title(
        f"Top-{top_k} {component.capitalize()} Eigenvalues Across Layers",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _plot_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    tokens: Optional[List[str]],
    title: str,
    cmap: str,
) -> None:
    """Helper to plot a single attention matrix as a heatmap."""
    vmax = max(abs(matrix.max()), abs(matrix.min()))
    if "RdBu" in cmap or "PiYG" in cmap:
        im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    else:
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11, fontweight="bold")

    if tokens is not None and len(tokens) <= 20:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
