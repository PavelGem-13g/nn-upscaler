"""Contrastive Unpaired Translation (CUT) loss modules.

Implements multi-layer PatchNCE (Noise Contrastive Estimation) loss
for maximizing mutual information between corresponding patches of input pixel art
and output realistic upscaled images without paired supervision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PatchNCELoss(nn.Module):
    """Patch-based Noise Contrastive Estimation (PatchNCE) loss.

    Formula:
        L_PatchNCE(v, v+, {v-}) = -log( exp(v · v+ / tau) / (exp(v · v+ / tau) + sum_k exp(v · v-_k / tau)) )

    Given normalized feature vectors from query (output) and key (input),
    the patch at the same spatial location is treated as positive (v+),
    while all other spatial locations in the image (and batch) are treated as negatives (v-).
    """

    def __init__(self, tau: float = 0.07, num_patches: int = 256):
        """
        Args:
            tau: Temperature hyperparameter scaling dot products (default: 0.07).
            num_patches: Number of spatial patch locations to sample per layer.
        """
        super().__init__()
        self.tau = float(tau)
        self.num_patches = int(num_patches)

    def _sample_and_normalize_patches(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract matching spatial patch vectors and L2-normalize them.

        Args:
            feat_q: Query features [B, C, H, W]
            feat_k: Key features [B, C, H, W]
            patch_ids: Optional pre-sampled spatial indices [num_patches]

        Returns:
            Tuple of:
                - q_flat: L2-normalized query vectors [B * S, C]
                - k_flat: L2-normalized key vectors [B * S, C]
                - patch_ids: sampled spatial indices
        """
        b, c, h, w = feat_q.shape
        num_spatial = h * w

        # Reshape to [B, C, H*W] -> [B, H*W, C]
        feat_q_flat = feat_q.view(b, c, -1).permute(0, 2, 1)
        feat_k_flat = feat_k.view(b, c, -1).permute(0, 2, 1)

        s_count = min(self.num_patches, num_spatial)

        if patch_ids is None:
            if s_count < num_spatial:
                patch_ids = torch.randperm(num_spatial, device=feat_q.device)[:s_count]
            else:
                patch_ids = torch.arange(num_spatial, device=feat_q.device)

        # Gather spatial patches [B, s_count, C]
        q_sampled = feat_q_flat[:, patch_ids, :]
        k_sampled = feat_k_flat[:, patch_ids, :]

        # Flatten batch and spatial patches: [B * s_count, C]
        q_norm = F.normalize(q_sampled.reshape(-1, c), p=2, dim=1)
        k_norm = F.normalize(k_sampled.reshape(-1, c), p=2, dim=1)

        return q_norm, k_norm, patch_ids

    def compute_single_layer_loss(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute PatchNCE cross-entropy loss for a single layer pair.

        Args:
            feat_q: Query features [B, C, H, W]
            feat_k: Key features [B, C, H, W]
            patch_ids: Optional patch indices to reuse across q and k

        Returns:
            Tuple of (loss_scalar, sampled_patch_ids)
        """
        q, k, p_ids = self._sample_and_normalize_patches(feat_q, feat_k, patch_ids=patch_ids)

        # Cross-correlation logits [N, N] where N = B * s_count
        # Diagonal (i == j) is positive pair (same patch location)
        # Off-diagonal (i != j) are negative pairs (other patches / images)
        logits = torch.matmul(q, k.t()) / self.tau

        labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)

        return loss, p_ids

    def forward(
        self,
        feats_q: torch.Tensor | list[torch.Tensor],
        feats_k: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute multi-layer PatchNCE loss.

        Args:
            feats_q: Query feature tensor or list of feature tensors.
            feats_k: Key feature tensor or list of feature tensors.

        Returns:
            Mean PatchNCE loss across all layers.
        """
        if isinstance(feats_q, torch.Tensor):
            feats_q = [feats_q]
        if isinstance(feats_k, torch.Tensor):
            feats_k = [feats_k]

        if len(feats_q) != len(feats_k):
            raise ValueError(
                f"Mismatch in number of layers: {len(feats_q)} query layers vs {len(feats_k)} key layers"
            )

        total_loss = torch.tensor(0.0, device=feats_q[0].device, dtype=torch.float32)

        for f_q, f_k in zip(feats_q, feats_k, strict=False):
            layer_loss, _ = self.compute_single_layer_loss(f_q, f_k)
            total_loss = total_loss + layer_loss

        return total_loss / max(1, len(feats_q))
