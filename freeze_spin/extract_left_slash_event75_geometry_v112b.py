from __future__ import annotations

"""v112b: correct the v112 rim observation model without weakening its gates.

v112 showed that fitting one ellipse to both Canny edges of the thick metal rim
inflates residuals even when the source overlay is correct.  This module keeps
all v112 target, cross-frame and fail-closed gates unchanged, but re-observes
the physical rim centreline from source colour pixels along normals to the
transfer-prior ellipse.  The transfer remains only a local search prior.
"""

import cv2
import numpy as np

from freeze_spin import extract_left_slash_event75_geometry_v112 as base


def _colour_centerline(image: np.ndarray, pred_desc: np.ndarray, *, samples: int = 240) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cx, cy, major, minor, angle = map(float, pred_desc)
    A, B = 0.5 * major, 0.5 * minor
    a = np.deg2rad(angle)
    ca, sa = float(np.cos(a)), float(np.sin(a))
    out = []

    for t in np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False):
        x, y = A * np.cos(t), B * np.sin(t)
        dx, dy = -A * np.sin(t), B * np.cos(t)
        px = cx + ca * x - sa * y
        py = cy + sa * x + ca * y
        tx = ca * dx - sa * dy
        ty = sa * dx + ca * dy
        norm = float(np.hypot(tx, ty))
        if norm < 1e-9:
            continue
        nx, ny = -ty / norm, tx / norm

        offsets = np.arange(-7.0, 7.0001, 0.5, dtype=np.float64)
        xs = px + offsets * nx
        ys = py + offsets * ny
        xi = np.rint(xs).astype(np.int32)
        yi = np.rint(ys).astype(np.int32)
        valid = (xi >= 0) & (xi < image.shape[1]) & (yi >= 0) & (yi < image.shape[0])
        if not np.any(valid):
            continue
        colours = hsv[yi[valid], xi[valid]]
        offs = offsets[valid]

        # The Utah rim is rendered as a strongly saturated red/orange tube in
        # these immutable source frames.  Include hue wraparound to avoid a
        # fragile 0/179 boundary.  This is source evidence, not colour fill.
        is_rim = (
            ((colours[:, 0] <= 15) | (colours[:, 0] >= 165))
            & (colours[:, 1] >= 70)
            & (colours[:, 2] >= 60)
        )
        rr = np.sort(offs[is_rim])
        if not len(rr):
            continue

        # Split separate red clusters along the normal and select the cluster
        # nearest the transfer-prior centreline.  The midpoint of that physical
        # tube section estimates the rim-wire centreline rather than either
        # visible Canny boundary.
        cuts = np.where(np.diff(rr) > 1.5)[0] + 1
        clusters = np.split(rr, cuts)
        cluster = min(clusters, key=lambda c: abs(float(np.mean(c))))
        centre_offset = 0.5 * (float(cluster.min()) + float(cluster.max()))
        if abs(centre_offset) > 6.0:
            continue
        out.append([px + centre_offset * nx, py + centre_offset * ny])

    return np.asarray(out, dtype=np.float32)


def extract_rim(image: np.ndarray, predicted_samples: np.ndarray) -> dict | None:
    pred_desc = base.canonical_ellipse(predicted_samples)
    support0 = _colour_centerline(image, pred_desc)
    if len(support0) < 30:
        return None

    support = support0.copy()
    # Deterministic robust refinement.  Keep the original v112 <=2.5 px source
    # fit gate; improve the observation model instead of relaxing the threshold.
    for _ in range(4):
        try:
            desc = base.canonical_ellipse(support)
        except cv2.error:
            return None
        curve = base.ellipse_points(desc, 720)
        d = base.nearest_curve_distance(support.astype(np.float64), curve)
        keep = d <= 2.5
        if int(keep.sum()) < 30 or int(keep.sum()) == len(support):
            break
        support = support[keep].astype(np.float32)

    if len(support) < 12:
        return None
    try:
        desc = base.canonical_ellipse(support)
    except cv2.error:
        return None
    curve = base.ellipse_points(desc, 720)
    d_fit = base.nearest_curve_distance(support.astype(np.float64), curve)

    train = support[::2].astype(np.float32)
    held = support[1::2].astype(np.float32)
    if len(train) < 5 or len(held) < 5:
        return None
    try:
        train_desc = base.canonical_ellipse(train)
    except cv2.error:
        return None
    held_d = base.nearest_curve_distance(held.astype(np.float64), base.ellipse_points(train_desc, 720))

    return {
        "predicted_ellipse": pred_desc.tolist(),
        "source_ellipse": desc.tolist(),
        "source_support_mode": "red_orange_rim_tube_normal_midpoint_centreline",
        "source_support_count_initial": int(len(support0)),
        "source_support_count": int(len(support)),
        "source_fit_median_px": float(np.median(d_fit)),
        "source_fit_p95_px": float(np.percentile(d_fit, 95)),
        "heldout_count": int(len(held_d)),
        "heldout_median_px": float(np.median(held_d)),
        "heldout_p95_px": float(np.percentile(held_d, 95)),
        "center_shift_from_transfer_prior_px": float(np.linalg.norm(desc[:2] - pred_desc[:2])),
        "major_axis_shift_from_transfer_prior_px": abs(float(desc[2] - pred_desc[2])),
        "minor_axis_shift_from_transfer_prior_px": abs(float(desc[3] - pred_desc[3])),
        "angle_shift_from_transfer_prior_deg": base.angle_diff_deg(float(desc[4]), float(pred_desc[4])),
        "source_curve_samples_px": base.ellipse_points(desc, 24).tolist(),
        "source_edge_support_px": support.astype(np.float64).tolist(),
    }


# frame_result resolves extract_rim from the base module global namespace, so
# this replaces only the falsified rim-observation model.  All v112 selection,
# target, consistency and permission logic remains unchanged.
base.extract_rim = extract_rim


if __name__ == "__main__":
    base.main()
