"""Scan a music collection and produce a training-prep triage report + draft dataset JSON.

Usage:
    python prep.py --src ~/Music/MyCollection --out ../dataset

Reads embedded tags (BPM, initial key, genre, artist/title) via mutagen,
applies quality filters, and writes:

    <out>/triage_report.csv    every file found, with keep/skip verdict + reason
    <out>/dataset.draft.json   ACE-Step full-format JSON for the keepers
                               (captions are DRAFTS built from tags — review each!)

Nothing is copied or modified. Review the report, prune, then copy keepers
into dataset/audio/ yourself.
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from mutagen import File as MutagenFile

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aiff", ".aif"}

# Rekordbox/MIK write Camelot codes (8A) or standard keys (Am); normalize to ACE-Step style
CAMELOT = {
    "1A": "Ab minor", "2A": "Eb minor", "3A": "Bb minor", "4A": "F minor",
    "5A": "C minor", "6A": "G minor", "7A": "D minor", "8A": "A minor",
    "9A": "E minor", "10A": "B minor", "11A": "F# minor", "12A": "Db minor",
    "1B": "B major", "2B": "F# major", "3B": "Db major", "4B": "Ab major",
    "5B": "Eb major", "6B": "Bb major", "7B": "F major", "8B": "C major",
    "9B": "G major", "10B": "D major", "11B": "A major", "12B": "E major",
}

MIX_HINTS = re.compile(
    r"(dj[\s_-]?mix|mixtape|full[\s_-]?set|live[\s_-]?@|podcast|episode|radio[\s_-]?show)",
    re.IGNORECASE,
)


def first_tag(tags, *names):
    if tags is None:
        return None
    for name in names:
        try:
            val = tags.get(name)
        except (KeyError, ValueError):
            val = None
        if val:
            v = val[0] if isinstance(val, list) else val
            return str(v).strip() or None
    return None


def normalize_key(raw):
    if not raw:
        return None
    raw = raw.strip()
    if raw.upper() in CAMELOT:
        return CAMELOT[raw.upper()]
    m = re.fullmatch(r"([A-Ga-g][#b]?)\s*(m|min|minor|maj|major)?", raw)
    if m:
        note = m.group(1).upper().replace("B", "b") if len(m.group(1)) > 1 else m.group(1).upper()
        quality = "minor" if (m.group(2) or "").lower().startswith("m") and (m.group(2) or "").lower() not in ("maj", "major") else "major"
        return f"{note} {quality}"
    return raw  # keep as-is; flag for review


def scan(src: Path, min_seconds: float, max_seconds: float, min_kbps: int):
    rows, samples = [], []
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    print(f"found {len(files)} audio files under {src}")

    for i, p in enumerate(files):
        row = {"path": str(p), "verdict": "keep", "reason": "", "bpm": "", "key": "",
               "genre": "", "duration_s": "", "bitrate_kbps": ""}
        try:
            mf = MutagenFile(p, easy=True)
            info = MutagenFile(p)  # non-easy for format-specific frames
        except Exception as e:
            row.update(verdict="skip", reason=f"unreadable: {e}")
            rows.append(row)
            continue
        if mf is None or mf.info is None:
            row.update(verdict="skip", reason="unreadable")
            rows.append(row)
            continue

        dur = getattr(mf.info, "length", 0) or 0
        kbps = int(getattr(mf.info, "bitrate", 0) / 1000) if getattr(mf.info, "bitrate", 0) else None
        lossless = p.suffix.lower() in {".wav", ".flac", ".aiff", ".aif"}
        row["duration_s"] = f"{dur:.0f}"
        row["bitrate_kbps"] = kbps or ("lossless" if lossless else "")

        bpm_raw = first_tag(mf.tags, "bpm") or first_tag(getattr(info, "tags", None), "TBPM")
        try:
            bpm = round(float(str(bpm_raw))) if bpm_raw else None
        except ValueError:
            bpm = None
        key = normalize_key(
            first_tag(mf.tags, "initialkey", "key")
            or first_tag(getattr(info, "tags", None), "TKEY")
        )
        genre = first_tag(mf.tags, "genre")
        row.update(bpm=bpm or "", key=key or "", genre=genre or "")

        name = p.name
        if dur < min_seconds:
            row.update(verdict="skip", reason=f"too short ({dur:.0f}s)")
        elif dur > max_seconds or MIX_HINTS.search(name):
            row.update(verdict="skip", reason="looks like a DJ mix/set")
        elif not lossless and kbps and kbps < min_kbps:
            row.update(verdict="skip", reason=f"low bitrate ({kbps}kbps)")

        rows.append(row)
        if row["verdict"] == "keep":
            caption_bits = [b for b in [genre, f"{bpm} BPM" if bpm else None] if b]
            samples.append({
                "id": f"{i:04d}",
                "audio_path": f"./audio/{name}",
                "filename": name,
                "caption": ", ".join(caption_bits) or "REVIEW ME",
                "genre": genre or "",
                "lyrics": "[Instrumental]",
                "raw_lyrics": "",
                "formatted_lyrics": "",
                "bpm": bpm,
                "keyscale": key or "",
                "timesignature": "4",
                "duration": round(dur),
                "language": "en",
                "is_instrumental": True,
                "custom_tag": "zynarai",
                "labeled": False,   # flips to True after your manual review
                "prompt_override": None,
            })
    return rows, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="root of your collection")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "dataset")
    ap.add_argument("--min-seconds", type=float, default=120)
    ap.add_argument("--max-seconds", type=float, default=900)
    ap.add_argument("--min-kbps", type=int, default=192)
    args = ap.parse_args()

    rows, samples = scan(args.src.expanduser(), args.min_seconds, args.max_seconds, args.min_kbps)
    args.out.mkdir(parents=True, exist_ok=True)

    report = args.out / "triage_report.csv"
    with open(report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    draft = args.out / "dataset.draft.json"
    draft.write_text(json.dumps({
        "metadata": {
            "name": "electronic_lora_v1",
            "custom_tag": "zynarai",
            "tag_position": "prepend",
            "created_at": datetime.now().isoformat(),
            "num_samples": len(samples),
            "all_instrumental": True,
            "genre_ratio": 90,
        },
        "samples": samples,
    }, indent=2))

    kept = sum(1 for r in rows if r["verdict"] == "keep")
    with_bpm = sum(1 for r in rows if r["verdict"] == "keep" and r["bpm"])
    with_key = sum(1 for r in rows if r["verdict"] == "keep" and r["key"])
    print(f"\n{len(rows)} scanned | {kept} keep | {len(rows)-kept} skip")
    print(f"of keepers: {with_bpm} have BPM tags, {with_key} have key tags")
    print(f"report: {report}\ndraft json: {draft}")


if __name__ == "__main__":
    main()
