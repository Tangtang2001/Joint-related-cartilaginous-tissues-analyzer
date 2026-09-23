"""Batch command-line interface for the paper's hybrid clustering analysis."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2

from .core import analyze_masks


def _slice_id(path: Path) -> str:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        raise ValueError(f"Filename has no numeric slice identifier: {path.name}")
    return matches[-1]


def _index_by_slice(paths: list[Path], kind: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        slice_id = _slice_id(path)
        if slice_id in indexed:
            raise ValueError(
                f"Duplicate {kind} files for slice {slice_id}: "
                f"{indexed[slice_id].name}, {path.name}"
            )
        indexed[slice_id] = path
    return indexed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the two-stage chondrocyte hybrid clustering method."
    )
    parser.add_argument("--instance-mask-dir", type=Path, required=True)
    parser.add_argument("--cartilage-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instance-glob", default="*_cp_masks.png")
    parser.add_argument("--cartilage-glob", default="*.jpg")
    parser.add_argument("--eps", type=float, default=15.0)
    parser.add_argument("--min-samples", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    instance_files = sorted(args.instance_mask_dir.glob(args.instance_glob))
    cartilage_files = sorted(args.cartilage_mask_dir.glob(args.cartilage_glob))
    if not instance_files:
        raise ValueError("No instance masks matched --instance-glob")
    if not cartilage_files:
        raise ValueError("No cartilage masks matched --cartilage-glob")

    instance_by_slice = _index_by_slice(instance_files, "instance mask")
    cartilage_by_slice = _index_by_slice(cartilage_files, "cartilage mask")
    missing = sorted(set(instance_by_slice) - set(cartilage_by_slice))
    if missing:
        raise ValueError(f"Missing cartilage masks for slices: {', '.join(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir = args.output_dir / "cluster_masks"
    mask_output_dir.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []

    for slice_id, instance_path in sorted(instance_by_slice.items()):
        cartilage_path = cartilage_by_slice[slice_id]
        result, final_mask = analyze_masks(
            instance_path,
            cartilage_path,
            eps_px=args.eps,
            min_samples=args.min_samples,
        )
        output_mask = mask_output_dir / f"slice_{slice_id}_final_clusters.tif"
        if not cv2.imwrite(str(output_mask), final_mask):
            raise OSError(f"Could not write cluster mask: {output_mask}")

        records.append(
            {
                "slice_id": slice_id,
                "instance_mask": instance_path.name,
                "cartilage_mask": cartilage_path.name,
                "cell_count": result.cell_count,
                "candidate_cluster_count": result.candidate_cluster_count,
                "final_cluster_count": result.final_cluster_count,
                "clustered_cell_count": result.clustered_cell_count,
                "clustered_fraction": result.clustered_fraction,
                "cartilage_area_px2": result.cartilage_area_px2,
                "clusters_per_area_px2": result.clusters_per_area_px2,
                "clusters_per_1e6_px2": result.clusters_per_area_px2 * 1_000_000,
                "eps_px": result.eps_px,
                "min_samples": result.min_samples,
                "contact_connectivity": 8,
            }
        )

    csv_path = args.output_dir / "cluster_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Analyzed {len(records)} slices. Results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
