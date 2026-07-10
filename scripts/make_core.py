"""Build dataset.core.json — the v1 instrumental house/tech-house core.

Rules, applied in order to dataset.autocaption.json:
  1. FULL_VOCAL tracks are excluded (verse/chorus lyrical content that would
     need corrected transcripts; v1 is instrumental-only by decision).
  2. Genre gate: keep the house family (tech/bass/deep/funky/jackin/minimal
     house). Untagged tracks pass only if the lead artist is a known house
     act (UNTAGGED_HOUSE). Everything else (UKG, 140, DnB, dance-pop,
     melodic/prog, techno, live-band electronica) is out of v1.
  3. BPM gate 115-134.
  4. CHOP_HOOK tracks stay in, with the caption ending rewritten to declare
     the chopped vocal hook; is_instrumental remains true (loops, not lyrics).

Track classifications are Claude's draft knowledge of these specific
releases — verify during ear-review (core_review.csv, `vocal_status` column;
`review` means unheard/unknown: confirm it is truly instrumental).
"""

import csv
import json
from pathlib import Path

DATASET = Path(__file__).parent.parent / "dataset"

# Full lyrical vocal content -> excluded from v1 (substring match on filename)
FULL_VOCAL = [
    "Night Rider", "Heroes (we could be)", "Spektrum", "Back Seat",
    "Escape (John Summit Remix)", "I Got Nothing", "No Ending", "A Milli",
    "Move On Up", "Hide U", "Cola", "Dior", "Asking", "Somedays",
    "This Feeling", "All That You Need", "Tell Me (Extended Mix)",
    "Tell Me It_s True", "Leave Me Like This", "Talk To Me", "Heavy Heart",
    "Ease My Mind", "Turn off the Lights", "Your Love (Original Mix)",
    "Let_s Go Dancing", "Ain_t No Other Man", "Envision", "You & Me",
    "Human (feat", "Lost feat", "Self Love", "All Night Long", "Notorious",
    "Bitch, Don_t Kill My Vibe", "BLOW (WHITE GIRL",
]

# Chopped hooks / spoken loops -> keep, declare in caption
CHOP_HOOK = [
    "Take Me Away", "Hypnotize", "I Want Your Soul", "Milkshake", "Freak",
    "Crush", "Selecta", "Bongoloco", "Fine Night", "Deceiver",
    "You Little Beauty", "In Chicago", "Make Me Feel", "God Made Me Phunky",
    "XTC", "Losing Control", "Is U", "VIP Business", "Slow Down",
    "On the Corner", "Jealous", "Sometimes The Going",
    "Needle On The Record", "Hustlin", "Burnin", "When The Bass Kicks In",
    "Can_t Decide", "Pega", "Blow Ya Mind", "Dance To The Music", "miss me",
    "Turn it Up", "Careless", "Beg (", "Make Me (Franky", "Lights Out",
    "Shoes On Please",
]

CORE_GENRES = {
    "tech house", "house", "bass house", "deep house", "funky house",
    "jackin house", "minimal / deep tech",
}

# untagged files pass the genre gate only via a known house-scene lead artist
UNTAGGED_HOUSE = {
    "acraze", "alex wann", "armand van helden", "biscits", "bruno furlan",
    "chelina manuhutu", "chris lorenzo", "cloonee", "fisher (oz)",
    "fm stroemer", "green velvet", "john summit", "laidback luke",
    "md x-spress", "odd mob", "overmono", "solardo", "hatiras", "nesi (es)",
    "matteo dentone", "warner case", "mochakk", "finn", "dateless",
}

BPM_MIN, BPM_MAX = 115, 134


def classify(sample):
    name = sample["filename"]
    genre = (sample.get("genre") or "").strip().lower()
    lead_artist = name.split(" - ")[0].split(",")[0].strip().lower()
    bpm = sample.get("bpm") or 0

    if any(s in name for s in FULL_VOCAL):
        return "exclude", "full lyrical vocals (v2 candidate with transcripts)"
    if genre and genre not in CORE_GENRES:
        return "exclude", f"outside v1 core genre ({sample.get('genre')})"
    if not genre and lead_artist not in UNTAGGED_HOUSE:
        return "exclude", "untagged, artist outside house core"
    if not (BPM_MIN <= bpm <= BPM_MAX):
        return "exclude", f"BPM {bpm} outside {BPM_MIN}-{BPM_MAX}"
    if any(s in name for s in CHOP_HOOK):
        return "chops", ""
    return "instrumental", ""


def main():
    data = json.loads((DATASET / "dataset.autocaption.json").read_text())
    core, review_rows, excluded = [], [], []

    for sample in data["samples"]:
        status, reason = classify(sample)
        if status == "exclude":
            excluded.append((sample["filename"], reason))
            continue
        if status == "chops":
            sample["caption"] = sample["caption"].replace(
                "instrumental club track", "sparse chopped vocal hooks"
            ).replace("with vocal hooks", "sparse chopped vocal hooks")
        sample["is_instrumental"] = True
        sample["lyrics"] = "[Instrumental]"
        core.append(sample)
        review_rows.append({
            "filename": sample["filename"],
            "vocal_status": status if status == "chops" else (
                "instrumental" if status == "instrumental" else status
            ),
            "bpm": sample["bpm"],
            "key": sample["keyscale"],
            "genre": sample.get("genre", ""),
            "caption": sample["caption"],
        })

    data["samples"] = core
    data["metadata"]["name"] = "electronic_lora_v1_core"
    data["metadata"]["num_samples"] = len(core)
    data["metadata"]["all_instrumental"] = True
    (DATASET / "dataset.core.json").write_text(json.dumps(data, indent=2))

    with open(DATASET / "core_review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
        w.writeheader()
        w.writerows(review_rows)
    with open(DATASET / "core_excluded.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "reason"])
        w.writerows(excluded)

    n_chops = sum(1 for r in review_rows if r["vocal_status"] == "chops")
    print(f"core: {len(core)} tracks ({n_chops} with chopped hooks, "
          f"{len(core)-n_chops} instrumental)")
    print(f"excluded: {len(excluded)}")
    print("wrote dataset.core.json, core_review.csv, core_excluded.csv")


if __name__ == "__main__":
    main()
