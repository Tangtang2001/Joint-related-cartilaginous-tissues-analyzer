"""Core implementation of the two-stage clustering method used in the paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN


@dataclass(frozen=True)
class HybridClusteringResult:
    """Cell-level assignments and slice-level measurements."""

    cell_labels: np.ndarray
    centroids_xy: np.ndarray
    candidate_labels: np.ndarray
    final_labels: np.ndarray
    candidate_cluster_count: int
    final_cluster_count: int
    cartilage_area_px2: int
    eps_px: float
    min_samples: int

    @property
    def cell_count(self) -> int:
        return int(self.cell_labels.size)

    @property
    def clustered_cell_count(self) -> int:
        return int(np.count_nonzero(self.final_labels >= 0))

    @property
    def clustered_fraction(self) -> float:
        if self.cell_count == 0:
            return 0.0
        return self.clustered_cell_count / self.cell_count

    @property
    def clusters_per_area_px2(self) -> float:
        if self.cartilage_area_px2 <= 0:
            raise ValueError("Cartilage area must be positive")
        return self.final_cluster_count / self.cartilage_area_px2

    def final_cluster_mask(self, instance_mask: np.ndarray) -> np.ndarray:
        """Return a uint16 image where 0 is background and 1..N are final clusters."""

        max_label = int(instance_mask.max(initial=0))
        lookup = np.zeros(max_label + 1, dtype=np.uint16)
        for cell_label, cluster_label in zip(
            self.cell_labels, self.final_labels, strict=True
        ):
            if cluster_label >= 0:
                lookup[int(cell_label)] = int(cluster_label) + 1
        return lookup[instance_mask]


def load_instance_mask(path: Path) -> np.ndarray:
    """Load a non-negative, integer Cellpose instance-label image."""

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read instance mask: {path}")
    if mask.ndim == 3:
        if not np.all(mask == mask[..., :1]):
            raise ValueError(f"Instance mask must be single-channel: {path}")
        mask = mask[..., 0]
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Instance mask must contain integer labels: {path}")
    if np.any(mask < 0):
        raise ValueError(f"Instance mask contains negative labels: {path}")
    return mask


def extract_centroids(instance_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract positive Cellpose labels and their (x, y) centroids."""

    cell_labels = np.unique(instance_mask)
    cell_labels = cell_labels[cell_labels > 0]
    if cell_labels.size == 0:
        return cell_labels.astype(np.int64), np.empty((0, 2), dtype=float)

    centers_yx = ndimage.center_of_mass(
        np.ones(instance_mask.shape, dtype=np.uint8),
        labels=instance_mask,
        index=cell_labels,
    )
    centroids_xy = np.asarray([(x, y) for y, x in centers_yx], dtype=float)
    return cell_labels.astype(np.int64, copy=False), centroids_xy


def touching_label_pairs(instance_mask: np.ndarray) -> set[tuple[int, int]]:
    """Find labels sharing an edge or corner (8-neighbor direct contact)."""

    pairs: set[tuple[int, int]] = set()
    neighbor_views = (
        (instance_mask[:, :-1], instance_mask[:, 1:]),
        (instance_mask[:-1, :], instance_mask[1:, :]),
        (instance_mask[:-1, :-1], instance_mask[1:, 1:]),
        (instance_mask[1:, :-1], instance_mask[:-1, 1:]),
    )

    for left, right in neighbor_views:
        contact = (left > 0) & (right > 0) & (left != right)
        if not np.any(contact):
            continue
        raw_pairs = np.column_stack((left[contact], right[contact]))
        raw_pairs.sort(axis=1)
        for first, second in np.unique(raw_pairs, axis=0):
            pairs.add((int(first), int(second)))
    return pairs


def _connected_components(
    nodes: set[int], touching_pairs: set[tuple[int, int]]
) -> list[set[int]]:
    adjacency = {node: set() for node in nodes}
    for first, second in touching_pairs:
        if first in nodes and second in nodes:
            adjacency[first].add(second)
            adjacency[second].add(first)

    components: list[set[int]] = []
    unvisited = set(nodes)
    while unvisited:
        start = unvisited.pop()
        component = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            new_neighbors = adjacency[current] & unvisited
            unvisited.difference_update(new_neighbors)
            component.update(new_neighbors)
            stack.extend(new_neighbors)
        components.append(component)
    return components


def cluster_chondrocytes(
    instance_mask: np.ndarray,
    *,
    eps_px: float = 15.0,
    min_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run centroid DBSCAN, then retain physically connected cell groups."""

    if eps_px <= 0:
        raise ValueError("eps_px must be positive")
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")

    cell_labels, centroids_xy = extract_centroids(instance_mask)
    if cell_labels.size < min_samples:
        noise = np.full(cell_labels.size, -1, dtype=int)
        return cell_labels, centroids_xy, noise.copy(), noise

    candidate_labels = DBSCAN(eps=eps_px, min_samples=min_samples).fit_predict(
        centroids_xy
    )
    final_labels = np.full(cell_labels.size, -1, dtype=int)
    label_to_index = {
        int(cell_label): index for index, cell_label in enumerate(cell_labels)
    }
    contacts = touching_label_pairs(instance_mask)

    next_cluster = 0
    for candidate in sorted(set(candidate_labels) - {-1}):
        candidate_cells = {
            int(cell_label)
            for cell_label in cell_labels[candidate_labels == candidate]
        }
        for component in _connected_components(candidate_cells, contacts):
            if len(component) < min_samples:
                continue
            for cell_label in component:
                final_labels[label_to_index[cell_label]] = next_cluster
            next_cluster += 1

    return cell_labels, centroids_xy, candidate_labels, final_labels


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    floodfilled = binary.copy().astype(np.uint8)
    height, width = binary.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(floodfilled, flood_mask, (0, 0), 255)
    return binary | cv2.bitwise_not(floodfilled)


def cartilage_area_from_white_background(path: Path) -> int:
    """Measure cartilage using the white-background preprocessing used in the study."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read cartilage mask: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    white_background = np.where(image > 254, 255, 0).astype(np.uint8)
    closed_background = cv2.morphologyEx(
        white_background,
        cv2.MORPH_CLOSE,
        np.ones((8, 8), dtype=np.uint8),
    )
    cartilage = cv2.bitwise_not(closed_background)
    cartilage = np.pad(cartilage, 2, mode="constant", constant_values=0)
    cartilage = _fill_holes(cartilage)[2:-2, 2:-2]
    area = int(np.count_nonzero(cartilage))
    if area == 0:
        raise ValueError(f"Cartilage area is zero after preprocessing: {path}")
    return area


def analyze_masks(
    instance_mask_path: Path,
    cartilage_mask_path: Path,
    *,
    eps_px: float = 15.0,
    min_samples: int = 3,
) -> tuple[HybridClusteringResult, np.ndarray]:
    """Analyze one matched pair of instance and cartilage masks."""

    instance_mask = load_instance_mask(instance_mask_path)
    cell_labels, centroids, candidates, final = cluster_chondrocytes(
        instance_mask,
        eps_px=eps_px,
        min_samples=min_samples,
    )
    candidate_count = len(set(candidates) - {-1})
    final_count = len(set(final) - {-1})
    result = HybridClusteringResult(
        cell_labels=cell_labels,
        centroids_xy=centroids,
        candidate_labels=candidates,
        final_labels=final,
        candidate_cluster_count=candidate_count,
        final_cluster_count=final_count,
        cartilage_area_px2=cartilage_area_from_white_background(cartilage_mask_path),
        eps_px=eps_px,
        min_samples=min_samples,
    )
    return result, result.final_cluster_mask(instance_mask)
