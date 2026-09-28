from __future__ import annotations

import shutil
import tempfile
from uuid import uuid4
from pathlib import Path

import streamlit as st

from main import DEFAULT_CONFIG, process_video

st.set_page_config(page_title="MatchLens AI", layout="wide")
st.title("MatchLens AI")
st.caption("Upload a football match clip to run detection, tracking, and basic metrics.")

uploaded = st.file_uploader("Upload match video", type=["mp4"])
if uploaded is not None:
    work_dir = Path(tempfile.gettempdir()) / "matchlensai"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / f"input-{uuid4().hex}.mp4"
    output_path = work_dir / f"annotated-{uuid4().hex}.mp4"
    with input_path.open("wb") as f:
        uploaded.seek(0)
        shutil.copyfileobj(uploaded, f)

    try:
        with st.spinner("Processing video..."):
            summary = process_video(str(input_path), str(output_path), DEFAULT_CONFIG)
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        st.stop()

    if not output_path.exists():
        st.error("Processing completed but no annotated output video was created.")
        st.stop()

    st.success("Processing complete")
    st.video(str(output_path))
    st.subheader("Player Metrics")
    if summary:
        st.json(summary)
    else:
        st.info("No players were tracked in this clip.")
