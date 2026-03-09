"""
Fake Data Generator for Raman Tissue Classifier.

Generates synthetic Raman spectra data with proper schema for testing the pipeline.
Supports both wide and long format CSV files.

Schema Requirements:
- Wide format: Each row is a sample, spectral columns named as wavenumber values
- Long format: Columns: sample_id, wavenumber, intensity, label
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def generate_raman_spectrum(
    wn: np.ndarray,
    label: int,
    seed: int | None = None,
    noise_level: float = 0.05,
    baseline_level: float = 0.1,
) -> np.ndarray:
    """Generate a synthetic Raman spectrum with characteristic peaks.

    Args:
        wn: Wavenumber vector (cm⁻¹).
        label: Class label (0 or 1).
        seed: Random seed for reproducibility.
        noise_level: Standard deviation of Gaussian noise.
        baseline_level: Amplitude of baseline polynomial.

    Returns:
        Intensity vector with same length as wn.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize with baseline (polynomial)
    baseline = baseline_level * (1 + 0.001 * (wn - 1200) ** 2)
    intensity = baseline.copy()

    # Define characteristic Raman peaks for different tissue types
    # Peak positions (cm⁻¹) and relative intensities
    if label == 0:  # Healthy tissue
        peaks = [
            (620, 0.8),   # Phenylalanine ring breathing
            (785, 0.6),   # DNA/RNA backbone
            (1004, 1.0),  # Phenylalanine symmetric ring breathing
            (1128, 0.5),  # C-N stretching
            (1240, 0.4),  # Amide III
            (1445, 0.7),  # CH2 deformation
            (1660, 0.6),  # Amide I
            (1745, 0.3),  # C=O stretching
        ]
    else:  # Tumor tissue
        peaks = [
            (620, 0.6),   # Phenylalanine (reduced)
            (785, 0.9),   # DNA/RNA (elevated)
            (1004, 0.8),  # Phenylalanine
            (1128, 0.7),  # C-N stretching
            (1240, 0.6),  # Amide III
            (1337, 0.5),  # CH3CH2 wagging
            (1445, 0.9),  # CH2 deformation (elevated)
            (1585, 0.4),  # Phenylalanine ring breathing
            (1660, 0.8),  # Amide I (elevated)
            (1745, 0.5),  # C=O stretching
        ]

    # Add Gaussian peaks
    for peak_wn, peak_intensity in peaks:
        sigma = 15.0  # Peak width
        peak = peak_intensity * np.exp(-0.5 * ((wn - peak_wn) / sigma) ** 2)
        intensity += peak

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, size=len(wn))
    intensity += noise

    # Ensure non-negative intensities
    intensity = np.maximum(intensity, 0)

    return intensity


def generate_wide_format(
    n_samples: int,
    wn_min: float = 600.0,
    wn_max: float = 1800.0,
    wn_step: float = 2.0,
    seed: int = 42,
    balance: float = 0.5,
) -> pd.DataFrame:
    """Generate fake data in wide format.

    Schema:
        - Each row represents one sample
        - Columns: label, patient_id, 600.0, 602.0, 604.0, ..., 1800.0
        - Spectral columns are named as wavenumber values (float)

    Args:
        n_samples: Total number of samples to generate.
        wn_min: Minimum wavenumber (cm⁻¹).
        wn_max: Maximum wavenumber (cm⁻¹).
        wn_step: Wavenumber step size (cm⁻¹).
        seed: Random seed for reproducibility.
        balance: Proportion of class 1 samples (0.0 to 1.0).

    Returns:
        DataFrame with wide format schema.
    """
    np.random.seed(seed)

    # Generate wavenumber vector
    wn = np.arange(wn_min, wn_max + wn_step, wn_step)
    n_wn = len(wn)

    logger.info(f"Generating {n_samples} samples with {n_wn} wavenumber points")

    # Generate labels
    n_class1 = int(n_samples * balance)
    n_class0 = n_samples - n_class1
    labels = np.array([0] * n_class0 + [1] * n_class1)
    np.random.shuffle(labels)

    # Generate patient IDs (for group-based CV)
    n_patients = max(10, n_samples // 5)
    patient_ids = np.random.randint(1, n_patients + 1, size=n_samples)

    # Generate spectra
    data = {"label": labels, "patient_id": patient_ids}

    for i, w in enumerate(wn):
        intensities = []
        for j, label in enumerate(labels):
            spec_seed = seed + j * 1000 + i
            intensity = generate_raman_spectrum(
                np.array([w]), label, seed=spec_seed, noise_level=0.05
            )[0]
            intensities.append(intensity)
        data[f"{w:.1f}"] = intensities

    df = pd.DataFrame(data)

    # Reorder columns: label, patient_id, then wavenumbers
    wn_cols = [f"{w:.1f}" for w in wn]
    df = df[["label", "patient_id"] + wn_cols]

    logger.info(f"Generated wide format: {df.shape}")
    logger.info(f"Class distribution: Class 0: {n_class0}, Class 1: {n_class1}")
    logger.info(f"Unique patients: {len(np.unique(patient_ids))}")

    return df


def generate_long_format(
    n_samples: int,
    wn_min: float = 600.0,
    wn_max: float = 1800.0,
    wn_step: float = 2.0,
    seed: int = 42,
    balance: float = 0.5,
) -> pd.DataFrame:
    """Generate fake data in long format.

    Schema:
        - Each row represents one (sample_id, wavenumber) pair
        - Columns: sample_id, wavenumber, intensity, label

    Args:
        n_samples: Total number of samples to generate.
        wn_min: Minimum wavenumber (cm⁻¹).
        wn_max: Maximum wavenumber (cm⁻¹).
        wn_step: Wavenumber step size (cm⁻¹).
        seed: Random seed for reproducibility.
        balance: Proportion of class 1 samples (0.0 to 1.0).

    Returns:
        DataFrame with long format schema.
    """
    np.random.seed(seed)

    # Generate wavenumber vector
    wn = np.arange(wn_min, wn_max + wn_step, wn_step)
    n_wn = len(wn)

    logger.info(f"Generating {n_samples} samples with {n_wn} wavenumber points")

    # Generate labels
    n_class1 = int(n_samples * balance)
    n_class0 = n_samples - n_class1
    labels = np.array([0] * n_class0 + [1] * n_class1)
    np.random.shuffle(labels)

    # Build long format DataFrame
    rows = []
    for sample_idx in range(n_samples):
        label = labels[sample_idx]
        sample_id = f"sample_{sample_idx:04d}"

        # Generate full spectrum for this sample
        spec_seed = seed + sample_idx * 1000
        spectrum = generate_raman_spectrum(wn, label, seed=spec_seed)

        # Add rows for each wavenumber
        for wn_val, intensity in zip(wn, spectrum):
            rows.append({
                "sample_id": sample_id,
                "wavenumber": wn_val,
                "intensity": intensity,
                "label": label,
            })

    df = pd.DataFrame(rows)

    logger.info(f"Generated long format: {df.shape}")
    logger.info(f"Class distribution: Class 0: {n_class0}, Class 1: {n_class1}")
    logger.info(f"Unique samples: {len(df['sample_id'].unique())}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fake Raman spectra data for testing."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="wide",
        choices=["wide", "long", "both"],
        help="Output format: wide, long, or both",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--wn-min",
        type=float,
        default=600.0,
        help="Minimum wavenumber (cm⁻¹)",
    )
    parser.add_argument(
        "--wn-max",
        type=float,
        default=1800.0,
        help="Maximum wavenumber (cm⁻¹)",
    )
    parser.add_argument(
        "--wn-step",
        type=float,
        default=2.0,
        help="Wavenumber step size (cm⁻¹)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.5,
        help="Proportion of class 1 samples (0.0 to 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/fake",
        help="Output directory for generated files",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Generate data
    if args.format in ["wide", "both"]:
        logger.info("=== Generating WIDE format ===")
        df_wide = generate_wide_format(
            n_samples=args.n_samples,
            wn_min=args.wn_min,
            wn_max=args.wn_max,
            wn_step=args.wn_step,
            seed=args.seed,
            balance=args.balance,
        )
        wide_path = output_dir / "fake_data_wide.csv"
        df_wide.to_csv(wide_path, index=False)
        logger.info(f"Saved wide format to {wide_path}")

    if args.format in ["long", "both"]:
        logger.info("=== Generating LONG format ===")
        df_long = generate_long_format(
            n_samples=args.n_samples,
            wn_min=args.wn_min,
            wn_max=args.wn_max,
            wn_step=args.wn_step,
            seed=args.seed,
            balance=args.balance,
        )
        long_path = output_dir / "fake_data_long.csv"
        df_long.to_csv(long_path, index=False)
        logger.info(f"Saved long format to {long_path}")

    logger.info("Fake data generation complete!")


if __name__ == "__main__":
    main()
