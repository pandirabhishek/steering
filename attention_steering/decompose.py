"""
Symmetric-Antisymmetric decomposition of attention matrices.

Any square matrix A can be uniquely decomposed as A = S + K where:
  S = (A + A^T) / 2  — symmetric, real eigenvalues
  K = (A - A^T) / 2  — skew-symmetric, purely imaginary eigenvalues
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


class AttentionDecomposer:
    """Decompose attention matrices and manipulate their spectral components."""

    def decompose(
        self, attention: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose attention matrix into symmetric and antisymmetric parts.

        Args:
            attention: Square attention matrix of shape (..., n, n).
                       Supports batched inputs.

        Returns:
            (S, K) where S is symmetric and K is skew-symmetric.
        """
        A = self._to_tensor(attention).float()
        A_T = A.transpose(-2, -1)
        S = (A + A_T) / 2.0
        K = (A - A_T) / 2.0
        return S, K

    def reconstruct(self, S: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Reconstruct attention matrix from its symmetric and antisymmetric parts."""
        return S + K

    def eigenspectrum_symmetric(
        self, S: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute eigenvalues of the symmetric component.
        Guaranteed real for symmetric matrices.

        Returns:
            Real eigenvalues sorted in descending order, shape (..., n).
        """
        eigenvalues = torch.linalg.eigvalsh(S)
        return eigenvalues.flip(-1)

    def eigenspectrum_antisymmetric(
        self, K: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute eigenvalues of the antisymmetric component.
        For skew-symmetric matrices these are purely imaginary (±bi),
        returned as the imaginary parts (real floats ±b).

        Returns:
            Imaginary parts of eigenvalues sorted by magnitude descending, shape (..., n).
        """
        eigenvalues = torch.linalg.eigvals(K)
        imag_parts = eigenvalues.imag
        sorted_idx = imag_parts.abs().argsort(dim=-1, descending=True)
        return imag_parts.gather(-1, sorted_idx)

    def eigen_decompose_symmetric(
        self, S: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full eigen-decomposition of symmetric component.

        Returns:
            (eigenvalues, eigenvectors) — both real.
        """
        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        idx = eigenvalues.argsort(dim=-1, descending=True)
        eigenvalues = eigenvalues.gather(-1, idx)
        eigenvectors = eigenvectors.gather(-1, idx.unsqueeze(-2).expand_as(eigenvectors))
        return eigenvalues, eigenvectors

    def eigen_decompose_antisymmetric(
        self, K: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full eigen-decomposition of antisymmetric component.

        Returns:
            (eigenvalues, eigenvectors) — eigenvalues are complex.
        """
        eigenvalues, eigenvectors = torch.linalg.eig(K)
        sorted_idx = eigenvalues.imag.abs().argsort(dim=-1, descending=True)
        eigenvalues = eigenvalues.gather(-1, sorted_idx)
        eigenvectors = eigenvectors.gather(
            -1, sorted_idx.unsqueeze(-2).expand_as(eigenvectors)
        )
        return eigenvalues, eigenvectors

    def spectral_filter_symmetric(
        self, S: torch.Tensor, top_k: int
    ) -> torch.Tensor:
        """
        Low-rank approximation of S keeping only top-k eigenvalues (by magnitude).
        """
        eigenvalues, eigenvectors = self.eigen_decompose_symmetric(S)
        mask = torch.zeros_like(eigenvalues)
        mask[..., :top_k] = 1.0
        filtered = eigenvalues * mask
        return eigenvectors @ torch.diag_embed(filtered) @ eigenvectors.transpose(-2, -1)

    def spectral_filter_antisymmetric(
        self, K: torch.Tensor, top_k: int
    ) -> torch.Tensor:
        """
        Low-rank approximation of K keeping only top-k eigenvalue pairs (by magnitude).
        """
        eigenvalues, eigenvectors = self.eigen_decompose_antisymmetric(K)
        mask = torch.zeros_like(eigenvalues.real)
        mask[..., :top_k] = 1.0
        filtered = eigenvalues * mask.to(eigenvalues.dtype)
        V_inv = torch.linalg.inv(eigenvectors)
        reconstructed = eigenvectors @ torch.diag_embed(filtered) @ V_inv
        return reconstructed.real

    def scale_symmetric_eigenvalues(
        self, S: torch.Tensor, scale: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """Scale eigenvalues of the symmetric component by a factor."""
        eigenvalues, eigenvectors = self.eigen_decompose_symmetric(S)
        scaled = eigenvalues * scale
        return eigenvectors @ torch.diag_embed(scaled) @ eigenvectors.transpose(-2, -1)

    def scale_antisymmetric_eigenvalues(
        self, K: torch.Tensor, scale: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """Scale eigenvalues of the antisymmetric component by a factor."""
        eigenvalues, eigenvectors = self.eigen_decompose_antisymmetric(K)
        scaled = eigenvalues * scale
        V_inv = torch.linalg.inv(eigenvectors)
        reconstructed = eigenvectors @ torch.diag_embed(scaled) @ V_inv
        return reconstructed.real

    def steer(
        self,
        attention: Union[torch.Tensor, np.ndarray],
        symmetric_scale: float = 1.0,
        antisymmetric_scale: float = 1.0,
        symmetric_top_k: Optional[int] = None,
        antisymmetric_top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        One-shot steering: decompose, modify spectra, reconstruct.

        Args:
            attention: Attention matrix (..., n, n).
            symmetric_scale: Multiplicative factor for symmetric eigenvalues.
            antisymmetric_scale: Multiplicative factor for antisymmetric eigenvalues.
            symmetric_top_k: If set, keep only top-k symmetric eigenvalues.
            antisymmetric_top_k: If set, keep only top-k antisymmetric eigenvalues.
        """
        S, K = self.decompose(attention)

        if symmetric_top_k is not None:
            S = self.spectral_filter_symmetric(S, symmetric_top_k)
        if symmetric_scale != 1.0:
            S = self.scale_symmetric_eigenvalues(S, symmetric_scale)

        if antisymmetric_top_k is not None:
            K = self.spectral_filter_antisymmetric(K, antisymmetric_top_k)
        if antisymmetric_scale != 1.0:
            K = self.scale_antisymmetric_eigenvalues(K, antisymmetric_scale)

        return self.reconstruct(S, K)

    def asymmetry_score(
        self, attention: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Ratio of antisymmetric energy to total energy.
        Values near 0 → nearly symmetric attention.
        Values near 1 → highly directional attention.
        """
        S, K = self.decompose(attention)
        sym_energy = (S ** 2).sum(dim=(-2, -1))
        asym_energy = (K ** 2).sum(dim=(-2, -1))
        total = sym_energy + asym_energy + 1e-10
        return asym_energy / total

    @staticmethod
    def _to_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x
