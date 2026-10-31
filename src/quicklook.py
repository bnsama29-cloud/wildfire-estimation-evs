from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_prep import OUTPUT_PATH

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence"
HISTOGRAM_BINS = np.logspace(-2, 4, 121)
MAP_SAMPLE_SIZE = 50_000
CHUNK_SIZE = 250_000


def generate_quicklook(
    data_path: str | Path = OUTPUT_PATH,
    evidence_dir: str | Path = EVIDENCE_DIR,
    map_sample_size: int = MAP_SAMPLE_SIZE,
) -> dict[str, int]:
    """Create the FRP histogram, monthly counts, and sampled India fire map."""
    source = Path(data_path)
    destination = Path(evidence_dir)
    if not source.exists():
        raise FileNotFoundError(f"Cleaned FIRMS dataset not found: {source}")
    if map_sample_size <= 0:
        raise ValueError("map_sample_size must be greater than zero")

    histogram = np.zeros(len(HISTOGRAM_BINS) - 1, dtype=np.int64)
    month_counts: dict[str, int] = {}
    sample_keys = np.empty(0, dtype=np.float64)
    sample_points = pd.DataFrame(columns=["lat", "lon", "frp"])
    row_count = 0
    rng = np.random.default_rng(42)
    for chunk in pd.read_csv(
        source,
        usecols=["lat", "lon", "acq_date", "frp"],
        dtype={"lat": "float32", "lon": "float32", "frp": "float32"},
        chunksize=CHUNK_SIZE,
    ):
        chunk["frp"] = pd.to_numeric(chunk["frp"], errors="coerce")
        valid = chunk["frp"].gt(0) & chunk["lat"].notna() & chunk["lon"].notna()
        chunk = chunk.loc[valid]
        if chunk.empty:
            continue
        row_count += len(chunk)
        values = chunk["frp"].to_numpy(dtype=np.float64)
        histogram += np.histogram(values, bins=HISTOGRAM_BINS)[0]
        months = pd.to_datetime(chunk["acq_date"], errors="coerce").dt.to_period("M")
        month_counts_chunk = months.dropna().astype(str).value_counts()
        for month, count in month_counts_chunk.items():
            month_counts[month] = month_counts.get(month, 0) + int(count)

        keys = rng.random(len(chunk))
        combined_keys = np.concatenate((sample_keys, keys))
        combined_points = pd.concat(
            (sample_points, chunk[["lat", "lon", "frp"]]), ignore_index=True
        )
        keep = min(map_sample_size, len(combined_keys))
        if len(combined_keys) > keep:
            selected = np.argpartition(combined_keys, -keep)[-keep:]
            sample_keys = combined_keys[selected]
            sample_points = combined_points.iloc[selected].reset_index(drop=True)
        else:
            sample_keys = combined_keys
            sample_points = combined_points

    if row_count == 0:
        raise ValueError(f"No valid positive-FRP records found in {source}")
    destination.mkdir(parents=True, exist_ok=True)

    _, axis = plt.subplots(figsize=(8, 5))
    axis.stairs(histogram, HISTOGRAM_BINS, fill=True, color="#e76f51", alpha=0.8)
    axis.set_xscale("log")
    axis.set(
        title="VIIRS fire radiative power distribution",
        xlabel="FRP (MW, logarithmic scale)",
        ylabel="Number of detections",
    )
    axis.grid(alpha=0.2, which="both")
    plt.tight_layout()
    plt.savefig(destination / "fig5_frp_hist.png", dpi=160)
    plt.close()

    monthly = pd.Series(month_counts, dtype="int64").sort_index()
    _, axis = plt.subplots(figsize=(10, 5))
    axis.plot(monthly.index, monthly.values, color="#2a9d8f", linewidth=1.5)
    axis.set(
        title="Monthly VIIRS fire detections",
        xlabel="Acquisition month",
        ylabel="Number of detections",
    )
    axis.tick_params(axis="x", labelrotation=60)
    axis.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(destination / "fig6_monthly.png", dpi=160)
    plt.close()

    _, axis = plt.subplots(figsize=(8, 8))
    points = axis.scatter(
        sample_points["lon"],
        sample_points["lat"],
        c=sample_points["frp"],
        s=4,
        alpha=0.6,
        cmap="inferno",
        norm=matplotlib.colors.LogNorm(vmin=max(float(sample_points["frp"].min()), 0.01)),
        rasterized=True,
    )
    axis.set(
        title=f"VIIRS fire detections across India (sample of {len(sample_points):,})",
        xlabel="Longitude",
        ylabel="Latitude",
        xlim=(68.0, 98.0),
        ylim=(6.0, 37.0),
        aspect="equal",
    )
    axis.grid(alpha=0.2)
    plt.colorbar(points, ax=axis, label="FRP (MW, logarithmic color scale)")
    plt.tight_layout()
    plt.savefig(destination / "fig9_map.png", dpi=160)
    plt.close()
    return {
        "valid_rows": row_count,
        "map_sample_rows": len(sample_points),
        "months": len(monthly),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the README quick-look figures.")
    parser.add_argument("--data", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE_DIR)
    args = parser.parse_args()
    result = generate_quicklook(args.data, args.output_dir)
    print(f"Generated quick-look figures from {result['valid_rows']:,} FIRMS rows.")
