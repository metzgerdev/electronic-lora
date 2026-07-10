"""Draft captions for every keeper in dataset.draft.json.

Fuses three signal sources into a caption skeleton per track:
  1. embedded genre/BPM/key tags (from prep.py's draft JSON)
  2. librosa features from a 60 s slice around each track's midpoint:
     tempo + key estimates (fills untagged files), brightness, sub-bass
     weight, percussive density, dynamic movement
  3. artist -> subgenre knowledge for well-known names, and a
     feat.-credit heuristic for vocals

Output:
    dataset/dataset.autocaption.json   captions filled, labeled stays false
    dataset/captions_review.csv        one row per track for ear-review

These are DRAFTS. Every caption still needs a listen before training.
"""

import csv
import json
import re
import sys
import warnings
from pathlib import Path

import librosa
import numpy as np

warnings.filterwarnings("ignore")

DATASET = Path(__file__).parent.parent / "dataset"
AUDIO_ROOT = Path.home() / "Documents/Code/training_music"

# artist -> (subgenre phrase, extra descriptor) for artists with a clear signature
ARTIST_STYLE = {
    "ac slater": ("bass house", "chunky wobbling bassline"),
    "adam port": ("afro house", "organic percussion and rolling groove"),
    "keinemusik": ("afro house", "organic percussion and rolling groove"),
    "alesso": ("progressive house", "big-room synth leads"),
    "artbat": ("melodic techno", "sweeping cinematic synths and driving groove"),
    "armand van helden": ("classic house", "filtered disco samples"),
    "basic channel": ("dub techno", "cavernous dub chords, hissy minimal groove"),
    "biscits": ("tech house", "bouncy bassline groove"),
    "borai & denham audio": ("uk garage breaks", "breakbeat drums and rave stabs"),
    "bruno furlan": ("tech house", "quirky percussion and punchy groove"),
    "camelphat": ("tech house", "dark rolling groove with melodic hooks"),
    "cassian": ("melodic house", "polished emotive synth work"),
    "cesco": ("140 deep dubstep", "weighty sub bass and sparse drums"),
    "hamdi": ("140 bass", "heavy sub wobble and UK bass energy"),
    "chris lorenzo": ("bass house", "prowling bassline and UK bass energy"),
    "four tet": ("house", "textured organic electronics"),
    "skrillex": ("bass-leaning house", "punchy sound design"),
    "hugel": ("latin tech house", "latin vocal chops and percussion"),
    "andruss": ("latin tech house", "latin percussion groove"),
    "chelina manuhutu": ("tech house", "driving percussive groove"),
    "anna lunoe": ("bass house", "club-ready bass groove"),
    "acraze": ("tech house", "vocal-chop hook and punchy drums"),
    "alex wann": ("house", "groovy sample-led house"),
    "ali love": ("melodic tech house", "hypnotic vocal hooks"),
    "taiki nulight": ("bass house", "gritty bassline pressure"),
    "franky rizardo": ("house", "sleek grooving bassline"),
    "tove lo": ("dance pop house", "pop vocal topline"),
}

GENRE_MAP = {
    "tech house": "tech house",
    "house": "house",
    "bass house": "bass house",
    "uk garage / bassline": "UK garage",
    "breaks / breakbeat / uk bass": "breakbeat UK bass",
    "afro house": "afro house",
    "melodic house & techno": "melodic house and techno",
    "techno (raw / deep / hypnotic)": "deep hypnotic techno",
    "techno (peak time / driving)": "peak-time driving techno",
    "140 / deep dubstep / grime": "140 deep dubstep",
    "deep house": "deep house",
    "drum & bass": "drum and bass",
    "funky house": "funky house",
    "dance / pop": "dance pop house",
}

FEAT_RE = re.compile(r"\bfeat\.|\bft\.", re.IGNORECASE)

KEYS = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(chroma_mean):
    best = (None, -2.0)
    for i in range(12):
        rolled = np.roll(chroma_mean, -i)
        for profile, quality in ((MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")):
            r = np.corrcoef(rolled, profile)[0, 1]
            if r > best[1]:
                best = (f"{KEYS[i]} {quality}", r)
    return best[0]


def analyze(path: Path):
    dur = librosa.get_duration(path=str(path))
    offset = max(0.0, dur / 2 - 30)
    y, sr = librosa.load(str(path), sr=22050, mono=True, offset=offset, duration=60)

    tempo = float(librosa.feature.tempo(y=y, sr=sr)[0])
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    key = estimate_key(chroma)

    S = np.abs(librosa.stft(y, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = S.sum() + 1e-9
    sub_ratio = S[freqs < 120].sum() / total
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    rms = librosa.feature.rms(y=y)[0]
    rms_var = float(np.std(rms) / (np.mean(rms) + 1e-9))
    onset_rate = len(librosa.onset.onset_detect(y=y, sr=sr)) / 60.0

    return {
        "tempo_est": round(tempo),
        "key_est": key,
        "sub_ratio": float(sub_ratio),
        "centroid": float(centroid),
        "rms_var": rms_var,
        "onset_rate": onset_rate,
    }


def parse_artists(filename: str):
    stem = Path(filename).stem
    artist_part = stem.split(" - ")[0] if " - " in stem else ""
    return [a.strip().lower() for a in artist_part.split(",")]


def compose_caption(sample, feats):
    artists = parse_artists(sample["filename"])
    genre_tag = (sample.get("genre") or "").strip().lower()

    genre_phrase, style_extra = None, None
    for a in artists:
        if a in ARTIST_STYLE:
            genre_phrase, style_extra = ARTIST_STYLE[a]
            break
    if genre_tag and genre_tag in GENRE_MAP:
        # explicit tag wins for the genre word; artist descriptor stays as extra
        genre_phrase = GENRE_MAP[genre_tag]
    if genre_phrase is None:
        genre_phrase = "house" if 118 <= feats["tempo_est"] <= 132 else "electronic"

    bits = []
    # energy/drive from tempo band
    bpm = sample.get("bpm") or feats["tempo_est"]
    if bpm >= 168:
        bits.append(f"fast rolling {genre_phrase}")
    elif bpm >= 136:
        bits.append(f"weighty {genre_phrase}")
    elif bpm >= 126:
        bits.append(f"driving {genre_phrase}")
    else:
        bits.append(f"groovy {genre_phrase}")

    if style_extra:
        bits.append(style_extra)
    if feats["sub_ratio"] > 0.45:
        bits.append("heavy sub bass")
    elif feats["sub_ratio"] > 0.33:
        bits.append("warm rounded low end")
    bits.append("bright crisp top end" if feats["centroid"] > 2600 else "dark moody tone")
    if feats["onset_rate"] > 5.5:
        bits.append("busy driving percussion")
    elif feats["onset_rate"] < 2.5:
        bits.append("sparse stripped-back drums")
    bits.append(
        "dynamic builds and drops" if feats["rms_var"] > 0.35 else "steady hypnotic groove"
    )

    has_vocals = bool(FEAT_RE.search(sample["filename"])) or genre_tag == "dance / pop"
    bits.append("with vocal hooks" if has_vocals else "instrumental club track")
    return ", ".join(bits), has_vocals


def main():
    draft = json.loads((DATASET / "dataset.draft.json").read_text())
    review_rows = []
    n = len(draft["samples"])
    for i, sample in enumerate(draft["samples"], 1):
        path = AUDIO_ROOT / sample["filename"]
        try:
            feats = analyze(path)
        except Exception as e:
            print(f"[{i}/{n}] FAILED {sample['filename']}: {e}", flush=True)
            continue

        caption, has_vocals = compose_caption(sample, feats)
        sample["caption"] = caption
        sample["is_instrumental"] = not has_vocals
        if not sample.get("bpm"):
            sample["bpm"] = feats["tempo_est"]
            bpm_src = "librosa (VERIFY)"
        else:
            bpm_src = "tag"
        if not sample.get("keyscale"):
            sample["keyscale"] = feats["key_est"]
            key_src = "librosa (VERIFY)"
        else:
            key_src = "tag"

        review_rows.append({
            "filename": sample["filename"],
            "bpm": sample["bpm"], "bpm_source": bpm_src,
            "key": sample["keyscale"], "key_source": key_src,
            "genre": sample.get("genre", ""),
            "vocals": "yes" if has_vocals else "no",
            "caption": caption,
        })
        print(f"[{i}/{n}] {sample['filename'][:60]}: {caption[:80]}", flush=True)

    (DATASET / "dataset.autocaption.json").write_text(json.dumps(draft, indent=2))
    with open(DATASET / "captions_review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
        w.writeheader()
        w.writerows(review_rows)
    print(f"\nwrote dataset.autocaption.json + captions_review.csv ({len(review_rows)} tracks)")


if __name__ == "__main__":
    main()
