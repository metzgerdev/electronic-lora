"""Contrastive re-captioning of the v1 core set.

Replaces the absolute-threshold feature phrases from autocaption.py with
dataset-relative ones: per feature axis, only the top and bottom quartile
of the 57 core tracks receive a descriptor; the middle 50% get none.
A phrase present in ~every caption teaches the model nothing — descriptors
should encode variance within the dataset, not genre constants (those are
absorbed by the trigger tag).

Re-analyzes audio (features weren't persisted last run), saves them to
core_features.json, rewrites dataset.core.json + core_review.csv in place.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from autocaption import AUDIO_ROOT, analyze  # noqa: E402

DATASET = Path(__file__).parent.parent / "dataset"

# phrases the old thresholds produced — stripped before re-composing
OLD_PHRASES = [
    "dynamic builds and drops", "steady hypnotic groove",
    "bright crisp top end", "dark moody tone",
    "busy driving percussion", "sparse stripped-back drums",
    "heavy sub bass", "warm rounded low end",
]
ENDINGS = ["sparse chopped vocal hooks", "instrumental club track"]

# feature axis -> (top-quartile phrase, bottom-quartile phrase)
AXES = {
    "rms_var": ("dramatic builds and drops", "steady rolling groove"),
    "centroid": ("bright crisp top end", "dark moody tone"),
    "onset_rate": ("busy layered percussion", "stripped-back minimal drums"),
    "sub_ratio": ("weighty sub-heavy low end", "lean tight low end"),
}


def main():
    data = json.loads((DATASET / "dataset.core.json").read_text())
    samples = data["samples"]

    feats_file = DATASET / "core_features.json"
    if feats_file.exists():
        feats = json.loads(feats_file.read_text())
    else:
        feats = {}
    for i, s in enumerate(samples, 1):
        if s["filename"] in feats:
            continue
        feats[s["filename"]] = analyze(AUDIO_ROOT / s["filename"])
        print(f"[{i}/{len(samples)}] analyzed {s['filename'][:60]}", flush=True)
    feats_file.write_text(json.dumps(feats, indent=2))

    # quartile cuts per axis across the core set
    cuts = {}
    for axis in AXES:
        vals = np.array([feats[s["filename"]][axis] for s in samples])
        cuts[axis] = (np.quantile(vals, 0.75), np.quantile(vals, 0.25))

    for s in samples:
        f = feats[s["filename"]]
        bits = [b.strip() for b in s["caption"].split(",")]
        ending = [b for b in bits if b in ENDINGS]
        bits = [b for b in bits if b not in OLD_PHRASES and b not in ENDINGS]

        for axis, (hi_phrase, lo_phrase) in AXES.items():
            hi, lo = cuts[axis]
            if f[axis] >= hi:
                bits.append(hi_phrase)
            elif f[axis] <= lo:
                bits.append(lo_phrase)

        s["caption"] = ", ".join(bits + ending)

    (DATASET / "dataset.core.json").write_text(json.dumps(data, indent=2))

    rows = list(csv.DictReader(open(DATASET / "core_review.csv")))
    cap_by_file = {s["filename"]: s["caption"] for s in samples}
    for r in rows:
        r["caption"] = cap_by_file.get(r["filename"], r["caption"])
    with open(DATASET / "core_review.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lengths = [len(s["caption"].split(",")) for s in samples]
    print(f"\nrewrote {len(samples)} captions "
          f"(clauses per caption: min {min(lengths)}, max {max(lengths)})")


if __name__ == "__main__":
    main()
