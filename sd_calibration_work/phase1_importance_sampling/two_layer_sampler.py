#!/usr/bin/env python3
"""
Two-Layer Importance Sampler for Mixed Training

Purpose:
1. First layer: Maintain real:sd ratio (default 0.8)
2. Second layer: Apply importance weights to SD subset

Usage:
    sampler = TwoLayerImportanceSampler(
        n_real=3200,
        n_sd=8960,
        sd_weights=weights,
        real_ratio=0.8,
        batch_size=32
    )
    dataloader = DataLoader(dataset, batch_sampler=sampler, ...)
"""

import torch
from torch.utils.data import Sampler, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from typing import List, Optional


class TwoLayerImportanceSampler(Sampler):
    """
    Two-layer sampler for mixed real+SD training.

    Layer 1: Choose between real and SD by ratio
    Layer 2: Within SD, apply weighted sampling

    Index mapping:
    - Real samples: [0, n_real-1]
    - SD samples: [n_real, n_real+n_sd-1]
    """

    def __init__(
        self,
        n_real: int,
        n_sd: int,
        sd_weights: np.ndarray,
        real_ratio: float = 0.8,
        batch_size: int = 32,
        generator: Optional[torch.Generator] = None
    ):
        """
        Args:
            n_real: Number of real samples
            n_sd: Number of SD samples
            sd_weights: Importance weights for SD samples (shape: [n_sd])
            real_ratio: Target ratio of real samples (0.8 = 80% real)
            batch_size: Batch size
            generator: Random generator for reproducibility
        """
        self.n_real = n_real
        self.n_sd = n_sd
        self.n_total = n_real + n_sd
        self.real_ratio = real_ratio
        self.batch_size = batch_size
        self.generator = generator

        # Validate inputs
        assert len(sd_weights) == n_sd, f"sd_weights length {len(sd_weights)} != n_sd {n_sd}"
        assert 0 < real_ratio < 1, f"real_ratio {real_ratio} must be in (0, 1)"

        # Create weighted sampler for SD subset
        self.sd_sampler = list(WeightedRandomSampler(
            weights=sd_weights,
            num_samples=len(sd_weights),
            replacement=True,
            generator=generator
        ))

        # Pre-allocate batch buffer
        self._buffer = []
        self._buffer_size = batch_size * 10  # Pre-generate 10 batches

        # Initialize buffer
        self._refill_buffer()

    def _refill_buffer(self):
        """Generate indices into buffer."""
        self._buffer = []
        n_real_batch = int(self.batch_size * self.real_ratio)
        n_sd_batch = self.batch_size - n_real_batch

        for _ in range(self._buffer_size // self.batch_size):
            # Sample real indices
            real_indices = torch.randint(
                low=0,
                high=self.n_real,
                size=(n_real_batch,),
                generator=self.generator
            ).tolist()

            # Sample SD indices using weighted sampler
            sd_indices_raw = []
            for _ in range(n_sd_batch):
                if len(self.sd_sampler) == 0:
                    # Refill SD sampler if exhausted
                    self.sd_sampler = list(WeightedRandomSampler(
                        weights=torch.ones(self.n_sd),  # Will be replaced
                        num_samples=self.n_sd,
                        replacement=True,
                        generator=self.generator
                    ))
                sd_idx = self.sd_sampler.pop(0)
                sd_indices_raw.append(sd_idx)

            # Offset SD indices to global index space
            sd_indices = [idx + self.n_real for idx in sd_indices_raw]

            # Combine and shuffle within batch
            batch_indices = real_indices + sd_indices
            perm = torch.randperm(len(batch_indices), generator=self.generator).tolist()
            batch_indices = [batch_indices[i] for i in perm]

            self._buffer.extend(batch_indices)

    def __iter__(self):
        """Iterate over batches."""
        while True:
            if len(self._buffer) < self.batch_size:
                self._refill_buffer()

            # Yield one batch
            batch = self._buffer[:self.batch_size]
            self._buffer = self._buffer[self.batch_size:]
            yield batch

    def __len__(self):
        """Return number of batches per epoch (approximate)."""
        return self.n_total // self.batch_size


def create_two_layer_dataloader(
    dataset,
    n_real: int,
    n_sd: int,
    sd_weights: np.ndarray,
    real_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 4,
    generator: Optional[torch.Generator] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with two-layer importance sampling.

    Args:
        dataset: Dataset instance (must be indexable)
        n_real: Number of real samples in dataset
        n_sd: Number of SD samples in dataset
        sd_weights: Importance weights for SD samples
        real_ratio: Target ratio of real samples
        batch_size: Batch size
        num_workers: Number of dataloader workers
        generator: Random generator
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader with two-layer sampling
    """
    sampler = TwoLayerImportanceSampler(
        n_real=n_real,
        n_sd=n_sd,
        sd_weights=sd_weights,
        real_ratio=real_ratio,
        batch_size=batch_size,
        generator=generator
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        **kwargs
    )


if __name__ == "__main__":
    # Test the sampler
    print("=" * 80)
    print("Two-Layer Importance Sampler Test")
    print("=" * 80)
    print()

    # Simulate dataset
    n_real = 3200
    n_sd = 8960
    batch_size = 32
    real_ratio = 0.8

    # Create dummy weights (higher for high S values)
    sd_weights = np.random.uniform(0.5, 5.0, size=n_sd)
    sd_weights[:n_sd//2] *= 0.5  # Lower weight for first half
    sd_weights[n_sd//2:] *= 2.0  # Higher weight for second half

    print(f"Dataset: {n_real} real + {n_sd} SD = {n_real + n_sd} total")
    print(f"Batch size: {batch_size}")
    print(f"Real ratio: {real_ratio}")
    print()

    # Create sampler
    sampler = TwoLayerImportanceSampler(
        n_real=n_real,
        n_sd=n_sd,
        sd_weights=sd_weights,
        real_ratio=real_ratio,
        batch_size=batch_size,
        generator=torch.Generator().manual_seed(42)
    )

    # Test sampling
    print("Testing sampling...")
    n_batches_to_test = 100
    real_count = 0
    sd_count = 0
    sd_high_weight_count = 0

    for i, batch in enumerate(sampler):
        if i >= n_batches_to_test:
            break

        for idx in batch:
            if idx < n_real:
                real_count += 1
            else:
                sd_count += 1
                sd_idx = idx - n_real
                if sd_idx >= n_sd // 2:
                    sd_high_weight_count += 1

    total_samples = real_count + sd_count
    actual_real_ratio = real_count / total_samples
    sd_high_weight_ratio = sd_high_weight_count / sd_count if sd_count > 0 else 0

    print(f"Sampled {n_batches_to_test} batches ({total_samples} samples)")
    print(f"  Real: {real_count} ({actual_real_ratio:.2%})")
    print(f"  SD: {sd_count} ({1-actual_real_ratio:.2%})")
    print(f"  SD high-weight: {sd_high_weight_count} ({sd_high_weight_ratio:.2%})")
    print()

    print("✓ Sampler test complete!")
