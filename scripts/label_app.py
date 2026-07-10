"""Streamlit click-to-label app for caption datasets.

    .venv/bin/streamlit run scripts/label_app.py

Reusable by design: everything vocabulary-specific lives in a schema JSON
(sidebar-configurable), so future datasets (e.g. the Zynar v2 set) need a
new schema file, not new code. The dataset JSON (ACE-Step full format) is
the source of truth; Save writes it directly (with a .bak of the previous
state per session).

Existing captions are parsed against the schema: clauses that match schema
options become selections; anything unrecognized is preserved verbatim as
"custom clauses" so hand-written detail is never lost.
"""

import json
import shutil
import time
from pathlib import Path

import streamlit as st

# ---------- pure logic (no streamlit) ----------------------------------------


def load_schema(path):
    return json.loads(Path(path).read_text())


def parse_caption(caption, schema):
    """caption -> (selections {slot_id: str|list}, custom_clauses [str])"""
    clauses = [c.strip() for c in caption.split(",") if c.strip()]
    selections = {s["id"]: ([] if s["select"] == "many" else None) for s in schema["slots"]}
    custom = []

    # space-joined prefix slots (e.g. "driving tech house" = energy + genre)
    space_slots = [s for s in schema["slots"] if s.get("join") == "space"]
    if clauses and space_slots:
        first = clauses[0]
        for s in space_slots:
            for opt in s["options"]:
                if first.startswith(opt + " "):
                    selections[s["id"]] = opt
                    first = first[len(opt) + 1:]
        clauses[0] = first

    option_index = {}
    for s in schema["slots"]:
        if s.get("join") == "space":
            continue
        for opt in s["options"]:
            option_index[opt] = (s["id"], s["select"])

    for clause in clauses:
        hit = option_index.get(clause)
        if hit is None:
            custom.append(clause)
        elif hit[1] == "many":
            selections[hit[0]].append(clause)
        else:
            selections[hit[0]] = clause
    return selections, custom


def compose_caption(selections, custom, schema):
    """selections + custom clauses -> caption in schema slot order.

    Clause order is *normalized* to the schema anatomy regardless of input
    order. Custom clauses are inserted after the slot named by the schema's
    top-level `custom_after` (default: before the last slot).
    """
    anchor = schema.get("custom_after")
    parts, prefix, inserted = [], "", False
    for s in schema["slots"]:
        sel = selections.get(s["id"])
        if s.get("join") == "space":
            if sel:
                prefix += sel + " "
            continue
        if prefix:
            parts.append((prefix + (sel or "")).rstrip())
            prefix = ""
        elif sel:
            parts.extend(sel if isinstance(sel, list) else [sel])
        if s["id"] == anchor:
            parts += custom
            inserted = True
    if not inserted:
        parts = parts[:-1] + custom + parts[-1:] if parts else list(custom)
    return ", ".join(p for p in parts if p)


# ---------- app ---------------------------------------------------------------

st.set_page_config(page_title="Track Labeler", layout="wide")

PROJECT = Path(__file__).parent.parent
with st.sidebar:
    st.header("Config")
    dataset_path = Path(st.text_input("Dataset JSON", str(PROJECT / "dataset/dataset.core.json")))
    schema_path = Path(st.text_input("Vocab schema", str(PROJECT / "dataset/vocab_schema.json")))
    audio_root = Path(st.text_input("Audio root", str(Path.home() / "Documents/Code/training_music")))
    only_unlabeled = st.toggle("Show unlabeled only", value=False)

if "data" not in st.session_state or st.session_state.get("_src") != str(dataset_path):
    st.session_state.data = json.loads(dataset_path.read_text())
    st.session_state._src = str(dataset_path)
    st.session_state._backed_up = False
    st.session_state.idx = 0

schema = load_schema(schema_path)
data = st.session_state.data
samples = data["samples"]
visible = [i for i, s in enumerate(samples) if not (only_unlabeled and s.get("labeled"))]
if not visible:
    st.success("All tracks labeled 🎉")
    st.stop()

n_done = sum(1 for s in samples if s.get("labeled"))
st.sidebar.progress(n_done / len(samples), text=f"{n_done}/{len(samples)} labeled")

# ---- navigation
if st.session_state.idx not in visible:
    st.session_state.idx = visible[0]
pos = visible.index(st.session_state.idx)
c_prev, c_pick, c_next = st.columns([1, 6, 1])
if c_prev.button("← prev", use_container_width=True) and pos > 0:
    st.session_state.idx = visible[pos - 1]
    st.rerun()
if c_next.button("next →", use_container_width=True) and pos < len(visible) - 1:
    st.session_state.idx = visible[pos + 1]
    st.rerun()
picked = c_pick.selectbox(
    "Track", visible, index=pos,
    format_func=lambda i: f"{'✅' if samples[i].get('labeled') else '⬜'} {samples[i]['filename']}",
    label_visibility="collapsed",
)
if picked != st.session_state.idx:
    st.session_state.idx = picked
    st.rerun()

idx = st.session_state.idx
track = samples[idx]
K = f"t{idx}_"  # per-track widget key prefix

# ---- player + metadata
audio_file = audio_root / track["filename"]
if audio_file.exists():
    st.audio(str(audio_file))
else:
    st.warning(f"audio not found: {audio_file}")

m1, m2, m3, m4 = st.columns(4)
bpm = m1.number_input("BPM", value=int(track.get("bpm") or 0), key=K + "bpm")
key_in = m2.text_input("Key", value=track.get("keyscale") or "", key=K + "key")
m3.text_input("Genre tag", value=track.get("genre") or "", disabled=True)
labeled = m4.checkbox("labeled (signed off)", value=bool(track.get("labeled")), key=K + "labeled")

# ---- schema-driven label pickers
selections, custom = parse_caption(track["caption"], schema)
new_sel = {}
for slot in schema["slots"]:
    opts = slot["options"]
    if slot["select"] == "many":
        new_sel[slot["id"]] = st.pills(
            slot["label"], opts, selection_mode="multi",
            default=[o for o in selections[slot["id"]] if o in opts], key=K + slot["id"],
        )
    else:
        cur = selections[slot["id"]]
        new_sel[slot["id"]] = st.pills(
            slot["label"], opts, selection_mode="single",
            default=cur if cur in opts else None, key=K + slot["id"],
        )

custom_text = st.text_input(
    "Custom clauses (comma-separated — preserved verbatim; promote recurring ones to the schema)",
    value=", ".join(custom), key=K + "custom",
)
new_custom = [c.strip() for c in custom_text.split(",") if c.strip()]

caption = compose_caption(new_sel, new_custom, schema)
st.markdown(f"**Caption preview:**  \n`{caption}`")

missing = [s["label"] for s in schema["slots"] if s.get("required") and not new_sel.get(s["id"])]
if missing:
    st.warning("required: " + "; ".join(missing))

# ---- save
s1, s2 = st.columns([1, 1])
if s1.button("💾 Save", type="primary", use_container_width=True, disabled=bool(missing)):
    track["caption"] = caption
    track["bpm"] = int(bpm) or track.get("bpm")
    track["keyscale"] = key_in.strip()
    track["labeled"] = labeled
    if not st.session_state._backed_up:
        shutil.copy(dataset_path, dataset_path.with_suffix(".json.bak"))
        st.session_state._backed_up = True
    data["metadata"]["num_samples"] = len(samples)
    dataset_path.write_text(json.dumps(data, indent=2))
    st.toast(f"saved: {track['filename'][:50]}")

if s2.button("💾 Save + next →", use_container_width=True, disabled=bool(missing)):
    track["caption"] = caption
    track["bpm"] = int(bpm) or track.get("bpm")
    track["keyscale"] = key_in.strip()
    track["labeled"] = True
    if not st.session_state._backed_up:
        shutil.copy(dataset_path, dataset_path.with_suffix(".json.bak"))
        st.session_state._backed_up = True
    dataset_path.write_text(json.dumps(data, indent=2))
    if pos < len(visible) - 1:
        st.session_state.idx = visible[pos + 1]
    st.rerun()
