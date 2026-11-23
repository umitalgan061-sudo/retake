"""Reference-aware source selection for the Retake face swapper.

All 25 images are always verified before a swap. Using all of their averaged
embeddings, however, need not be the best way to reproduce an individual
face in a given head pose: some photos are blurry, strongly turned or have
different facial hair. Prepare a small *set of candidate source pools* and
let the same ArcFace verifier evaluate the actual generated results.

There are no guessed absolute identity thresholds and no reference is
mistakenly treated as a different person merely because of a turned head.
"""
from __future__ import annotations

import math
from typing import Any, Sequence


def _weighted(indices: Sequence[int], weights: Sequence[int]) -> tuple[int, ...]:
    return tuple(index for index in indices for _ in range(int(weights[index])))


def plan_identity_source_sets(
    weights: Sequence[int],
    identity_report: dict[str, Any],
    pose_distances: Sequence[float | None] | None = None,
    *,
    consensus_size: int = 12,
    pose_size: int = 8
) -> list[tuple[str, tuple[int, ...]]]:
    """Return stable source pools: all / consensus / target-pose matched.

    The full set is always first. Additional pools require evidence:
    per-reference ArcFace agreement for consensus and valid mesh descriptor
    distances for pose. The best actual *output* is chosen later by ArcFace.
    """
    count = len(weights)
    if count == 0 or any(type(value) is not int or value < 1 or value > 3 for value in weights):
        raise ValueError("Referans ağırlıkları 1..3 aralığında olmalı.")
    full = _weighted(range(count), weights)
    pools: list[tuple[str, tuple[int, ...]]] = [("all-25", full)]
    medians = identity_report.get("reference_median_similarities")
    if (
        not isinstance(medians, list) or len(medians) != count
        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in medians)
    ):
        return pools

    excluded = set(identity_report.get("outlier_indices", ()))
    accepted = [i for i in range(count) if i not in excluded]
    # A coherent subset should contain enough diverse examples to avoid
    # turning one distinctive expression / age into the universal identity.
    if len(accepted) < 6:
        return pools

    # An outlier stays in the first all-photo attempt because the measured
    # output, not a heuristic, has the final say.  But it must not be allowed
    # to make "all 25" the only broad identity hypothesis.  A separate
    # trusted-full pool keeps the diversity of the profile (unlike a compact
    # consensus) while removing only references ArcFace already flagged.
    # This matters most for a sharp but visibly different beard, haircut or
    # lighting outlier: it can lift a global cosine score while making the
    # resulting face look less like the person to a human observer.
    trusted_full = _weighted(accepted, weights)
    if trusted_full != full:
        pools.append((f"trusted-{len(accepted)}", trusted_full))

    consensus_size = min(len(accepted), max(6, consensus_size))
    sorted_consensus = sorted(
        accepted, key=lambda i: (-float(medians[i]), -weights[i], i)
    )
    trusted = sorted(sorted_consensus[:consensus_size])
    consensus = _weighted(trusted, weights)
    if len(trusted) < count and consensus != full:
        pools.append(("consensus", consensus))

    if (
        not isinstance(pose_distances, (list, tuple))
        or len(pose_distances) != count
    ):
        return pools
    possible = [
        index for index in accepted
        if isinstance(pose_distances[index], (float, int))
        and not isinstance(pose_distances[index], bool)
        and math.isfinite(pose_distances[index])
        and pose_distances[index] >= 0
    ]
    if len(possible) < 6:
        return pools
    pose_size = min(len(possible), max(6, pose_size))
    # Prioritize expression / head orientation; if distances are nearly equal,
    # pick photos with the strongest agreement and image quality.
    ranked = sorted(
        possible, key=lambda i: (
            round(float(pose_distances[i]), 4),
            -float(medians[i]), -weights[i], i
        )
    )
    selected = sorted(ranked[:pose_size])
    pose_pool = _weighted(selected, weights)
    if pose_pool != full and all(pool != pose_pool for _, pool in pools):
        pools.append(("pose-match", pose_pool))
    return pools


def select_texture_reference(
    weights: Sequence[int],
    identity_report: dict[str, Any],
    pose_distances: Sequence[float | None] | None = None,
) -> int | None:
    """Choose one real (not pixel-averaged) face for optional UniFace detail.

    It is a *candidate*, never the only appearance source. Weighted sharpness
    and self-consistency provide a frontal quality anchor; a similar target
    pose avoids copying a strongly turned face's beard / skin into the target.
    """
    count = len(weights)
    medians = identity_report.get("reference_median_similarities")
    if (
        not isinstance(medians, list) or len(medians) != count
        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in medians)
        or count < 6
    ):
        return None
    outliers = set(identity_report.get("outlier_indices", ()))
    accepted = [i for i in range(count) if i not in outliers]
    if not accepted:
        return None
    poses_ok = isinstance(pose_distances, (list, tuple)) and len(pose_distances) == count
    possible = [
        i for i in accepted if poses_ok and isinstance(pose_distances[i], (int, float))
        and not isinstance(pose_distances[i], bool)
        and math.isfinite(pose_distances[i]) and pose_distances[i] >= 0
    ]
    if len(possible) >= 6:
        # A pose-close but blurry photo is worse than a very sharp reference:
        # choose the best quality/identity anchor among the top eight poses.
        candidates = sorted(possible, key=lambda i: float(pose_distances[i]))[:8]
    else:
        candidates = accepted
    return max(candidates, key=lambda i: (weights[i], float(medians[i]), -i))
