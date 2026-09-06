from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def extract_mono(path: Path, sample_rate: int, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", str(output),
        ],
        check=True,
    )


def fft_lag(reference: np.ndarray, sample: np.ndarray, max_lag: int) -> tuple[int, float]:
    reference = reference.astype(np.float64)
    sample = sample.astype(np.float64)
    reference -= reference.mean()
    sample -= sample.mean()
    fft_size = 1 << ((len(reference) + len(sample) - 2).bit_length())
    correlation = np.fft.irfft(
        np.fft.rfft(sample, fft_size) * np.conj(np.fft.rfft(reference, fft_size)),
        fft_size,
    )
    correlation = np.concatenate(
        (correlation[-(len(reference) - 1):], correlation[:len(sample)])
    )
    lags = np.arange(-len(reference) + 1, len(sample))
    allowed = np.abs(lags) <= max_lag
    allowed_correlation = correlation[allowed]
    best = int(np.argmax(allowed_correlation))
    lag = int(lags[allowed][best])
    normalized_peak = float(
        allowed_correlation[best]
        / (np.linalg.norm(reference) * np.linalg.norm(sample) + 1e-12)
    )
    return lag, normalized_peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--reference", default="Broadcast")
    parser.add_argument("--sample-rate", type=int, default=4000)
    parser.add_argument("--max-lag-seconds", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    by_label = {row["label"]: row for row in config["angles"]}
    if args.reference not in by_label:
        raise ValueError(f"Unknown reference angle: {args.reference}")

    with tempfile.TemporaryDirectory(prefix="nba-angle-audio-") as temporary:
        temporary_path = Path(temporary)
        audio = {}
        for label, row in by_label.items():
            raw = temporary_path / f"{len(audio):02d}.s16"
            extract_mono(args.clips / row["file"], args.sample_rate, raw)
            audio[label] = np.fromfile(raw, dtype="<i2")

        reference = audio[args.reference]
        results = []
        for label in by_label:
            lag_samples, peak = fft_lag(
                reference,
                audio[label],
                int(args.max_lag_seconds * args.sample_rate),
            )
            results.append(
                {
                    "label": label,
                    "lag_seconds_vs_reference": round(lag_samples / args.sample_rate, 4),
                    "normalized_correlation_peak": round(peak, 4),
                    "confidence": "high" if peak >= 0.75 else "moderate" if peak >= 0.4 else "low",
                }
            )

    payload = {
        "reference_angle": args.reference,
        "sample_rate_hz": args.sample_rate,
        "interpretation": "positive lag means the angle's corresponding action/audio occurs later than the reference timeline",
        "offsets": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
