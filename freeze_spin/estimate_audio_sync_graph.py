from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def extract_mono(path: Path, sample_rate: int, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-af", "highpass=f=120,lowpass=f=3800",
            "-f", "s16le", str(output),
        ],
        check=True,
    )


def robust_signal(raw: np.ndarray) -> np.ndarray:
    x = raw.astype(np.float64)
    x -= np.median(x)
    scale = np.percentile(np.abs(x), 95) + 1e-9
    x = np.clip(x / scale, -3.0, 3.0)
    # Emphasise arena transients and de-emphasise slow crowd/commentary energy.
    x = np.diff(x, prepend=x[0])
    x -= x.mean()
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x


def fft_lag(reference: np.ndarray, sample: np.ndarray, max_lag: int) -> tuple[int, float]:
    n = 1 << ((len(reference) + len(sample) - 2).bit_length())
    corr = np.fft.irfft(
        np.fft.rfft(sample, n) * np.conj(np.fft.rfft(reference, n)),
        n,
    )
    corr = np.concatenate((corr[-(len(reference) - 1):], corr[:len(sample)]))
    lags = np.arange(-len(reference) + 1, len(sample))
    allowed = np.abs(lags) <= max_lag
    values = corr[allowed]
    best_index = int(np.argmax(values))
    lag = int(lags[allowed][best_index])
    return lag, float(values[best_index])


def overlap_ncc(reference: np.ndarray, sample: np.ndarray, lag: int) -> float:
    if lag >= 0:
        a = reference[: max(0, min(len(reference), len(sample) - lag))]
        b = sample[lag : lag + len(a)]
    else:
        start = -lag
        a = reference[start : start + max(0, min(len(reference) - start, len(sample)))]
        b = sample[: len(a)]
    if len(a) < 100:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def weighted_solution(n_nodes: int, edges: list[dict], reference_index: int) -> tuple[np.ndarray, np.ndarray]:
    active = [i for i in range(n_nodes) if i != reference_index]
    col = {node: c for c, node in enumerate(active)}
    rows = []
    b = []
    base_weights = []
    for edge in edges:
        row = np.zeros(len(active), dtype=np.float64)
        i, j = edge["i"], edge["j"]
        if i != reference_index:
            row[col[i]] -= 1.0
        if j != reference_index:
            row[col[j]] += 1.0
        rows.append(row)
        b.append(edge["lag_seconds"])
        base_weights.append(max(edge["quality"], 0.03) ** 2)

    A = np.vstack(rows)
    bvec = np.asarray(b, dtype=np.float64)
    base = np.asarray(base_weights, dtype=np.float64)
    robust = np.ones_like(base)
    x = np.zeros(len(active), dtype=np.float64)

    for _ in range(8):
        weights = np.sqrt(base * robust)
        Aw = A * weights[:, None]
        bw = bvec * weights
        x, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        residual = A @ x - bvec
        scale = max(0.010, 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9)
        threshold = 2.5 * scale
        robust = np.ones_like(base)
        bad = np.abs(residual) > threshold
        robust[bad] = threshold / (np.abs(residual[bad]) + 1e-12)

    offsets = np.zeros(n_nodes, dtype=np.float64)
    for node in active:
        offsets[node] = x[col[node]]
    residual = np.asarray(
        [offsets[e["j"]] - offsets[e["i"]] - e["lag_seconds"] for e in edges],
        dtype=np.float64,
    )
    return offsets, residual


def ensure_connected_edges(n_nodes: int, candidate_edges: list[dict], threshold: float) -> list[dict]:
    chosen = [e for e in candidate_edges if e["quality"] >= threshold]
    incident = {i: [] for i in range(n_nodes)}
    for e in candidate_edges:
        incident[e["i"]].append(e)
        incident[e["j"]].append(e)
    covered = set()
    for e in chosen:
        covered.add(e["i"])
        covered.add(e["j"])
    for node in range(n_nodes):
        if node in covered:
            continue
        if incident[node]:
            best = max(incident[node], key=lambda e: e["quality"])
            if best not in chosen:
                chosen.append(best)
            covered.add(best["i"])
            covered.add(best["j"])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--reference", default="Broadcast")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--max-lag-seconds", type=float, default=2.0)
    parser.add_argument("--min-edge-quality", type=float, default=0.10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    angles = config["angles"]
    labels = [row["label"] for row in angles]
    if args.reference not in labels:
        raise ValueError(f"Unknown reference angle: {args.reference}")
    reference_index = labels.index(args.reference)
    reference_freeze = float(angles[reference_index]["freeze_time"])

    with tempfile.TemporaryDirectory(prefix="nba-sync-graph-") as temporary:
        root = Path(temporary)
        signals = []
        for index, row in enumerate(angles):
            raw_path = root / f"{index:02d}.s16"
            extract_mono(args.clips / row["file"], args.sample_rate, raw_path)
            raw = np.fromfile(raw_path, dtype="<i2")
            signals.append(robust_signal(raw))

        candidates = []
        max_lag = int(round(args.max_lag_seconds * args.sample_rate))
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                lag_samples, peak = fft_lag(signals[i], signals[j], max_lag)
                ncc = overlap_ncc(signals[i], signals[j], lag_samples)
                quality = max(0.0, ncc)
                candidates.append(
                    {
                        "i": i,
                        "j": j,
                        "from": labels[i],
                        "to": labels[j],
                        "lag_samples": lag_samples,
                        "lag_seconds": lag_samples / args.sample_rate,
                        "fft_peak": peak,
                        "quality": quality,
                    }
                )

    edges = ensure_connected_edges(len(labels), candidates, args.min_edge_quality)
    offsets, residuals = weighted_solution(len(labels), edges, reference_index)

    edge_rows = []
    for edge, residual in zip(edges, residuals):
        row = {k: v for k, v in edge.items() if k not in {"i", "j"}}
        row["lag_seconds"] = round(float(row["lag_seconds"]), 5)
        row["quality"] = round(float(row["quality"]), 4)
        row["graph_residual_seconds"] = round(float(residual), 5)
        edge_rows.append(row)

    angle_rows = []
    for index, row in enumerate(angles):
        incident = [
            (e, residuals[k]) for k, e in enumerate(edges)
            if e["i"] == index or e["j"] == index
        ]
        qualities = [e["quality"] for e, _ in incident]
        residual_abs = [abs(float(r)) for _, r in incident]
        mean_quality = float(np.mean(qualities)) if qualities else 0.0
        median_residual = float(np.median(residual_abs)) if residual_abs else math.inf
        confidence = (
            "high" if mean_quality >= 0.35 and median_residual <= 0.035
            else "moderate" if mean_quality >= 0.15 and median_residual <= 0.080
            else "low"
        )
        manual = float(row["freeze_time"])
        predicted = reference_freeze + float(offsets[index])
        angle_rows.append(
            {
                "label": row["label"],
                "offset_seconds_vs_reference": round(float(offsets[index]), 5),
                "predicted_freeze_time": round(predicted, 5),
                "manual_freeze_time_for_validation_only": round(manual, 5),
                "prediction_minus_manual_seconds": round(predicted - manual, 5),
                "incident_edge_count": len(incident),
                "mean_incident_quality": round(mean_quality, 4),
                "median_incident_graph_residual_seconds": (
                    None if not residual_abs else round(median_residual, 5)
                ),
                "confidence": confidence,
            }
        )

    payload = {
        "method": "pairwise transient-audio FFT correlation + robust weighted graph synchronization",
        "reference_angle": args.reference,
        "reference_freeze_time": reference_freeze,
        "sample_rate_hz": args.sample_rate,
        "max_lag_seconds": args.max_lag_seconds,
        "selected_edge_count": len(edges),
        "all_pair_count": len(candidates),
        "angles": angle_rows,
        "selected_edges": edge_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
