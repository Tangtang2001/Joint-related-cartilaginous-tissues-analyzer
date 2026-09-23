from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from chondrocyte_hybrid_clustering.cli import main
from chondrocyte_hybrid_clustering.core import (
    cartilage_area_from_white_background,
    cluster_chondrocytes,
    extract_centroids,
    touching_label_pairs,
)


def _three_touching_cells() -> np.ndarray:
    mask = np.zeros((30, 40), dtype=np.uint16)
    mask[10:15, 10:15] = 1
    mask[10:15, 15:20] = 2
    mask[10:15, 20:25] = 3
    return mask


def test_extract_centroids_uses_instance_labels() -> None:
    labels, centers = extract_centroids(_three_touching_cells())
    np.testing.assert_array_equal(labels, [1, 2, 3])
    np.testing.assert_allclose(centers, [[12, 12], [17, 12], [22, 12]])


def test_contact_detection_uses_eight_neighbors() -> None:
    mask = np.zeros((4, 4), dtype=np.uint16)
    mask[1, 1] = 1
    mask[2, 2] = 2
    assert touching_label_pairs(mask) == {(1, 2)}


def test_candidate_cluster_is_retained_when_all_cells_are_connected() -> None:
    _, _, candidates, final = cluster_chondrocytes(
        _three_touching_cells(), eps_px=15, min_samples=3
    )
    np.testing.assert_array_equal(candidates, [0, 0, 0])
    np.testing.assert_array_equal(final, [0, 0, 0])


def test_candidate_cluster_is_rejected_without_physical_contact() -> None:
    mask = np.zeros((30, 40), dtype=np.uint16)
    mask[10:13, 10:13] = 1
    mask[10:13, 15:18] = 2
    mask[10:13, 20:23] = 3
    _, _, candidates, final = cluster_chondrocytes(mask, eps_px=15, min_samples=3)
    np.testing.assert_array_equal(candidates, [0, 0, 0])
    np.testing.assert_array_equal(final, [-1, -1, -1])


def test_candidate_cluster_is_split_into_connected_components() -> None:
    mask = np.zeros((40, 50), dtype=np.uint16)
    for index, x in enumerate((3, 6, 9, 20, 23, 26), start=1):
        mask[10:13, x : x + 3] = index
    _, _, candidates, final = cluster_chondrocytes(mask, eps_px=15, min_samples=3)
    assert len(set(candidates)) == 1
    np.testing.assert_array_equal(final, [0, 0, 0, 1, 1, 1])


def test_cartilage_area_matches_white_background_preprocessing(tmp_path: Path) -> None:
    image = np.full((30, 40), 255, dtype=np.uint8)
    image[10:20, 10:30] = 100
    path = tmp_path / "mask_00001.png"
    assert cv2.imwrite(str(path), image)
    assert cartilage_area_from_white_background(path) == 200


def test_batch_cli_matches_files_by_slice_number(tmp_path: Path) -> None:
    instance_dir = tmp_path / "instances"
    cartilage_dir = tmp_path / "cartilage"
    output_dir = tmp_path / "output"
    instance_dir.mkdir()
    cartilage_dir.mkdir()
    assert cv2.imwrite(
        str(instance_dir / "cell_00007_cp_masks.png"), _three_touching_cells()
    )
    cartilage = np.full((30, 40), 255, dtype=np.uint8)
    cartilage[5:25, 5:35] = 100
    assert cv2.imwrite(str(cartilage_dir / "mask_00007.jpg"), cartilage)

    exit_code = main(
        [
            "--instance-mask-dir",
            str(instance_dir),
            "--cartilage-mask-dir",
            str(cartilage_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    csv_text = (output_dir / "cluster_metrics.csv").read_text(encoding="utf-8")
    assert "00007" in csv_text
    assert ",1," in csv_text
    assert (output_dir / "cluster_masks" / "slice_00007_final_clusters.tif").exists()
