from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from main import DEFAULT_CONFIG, process_video

st.set_page_config(page_title="MatchLens AI", layout="wide")
st.title("MatchLens AI")
st.caption("Upload a football match clip to run detection, tracking, and basic metrics.")

uploaded = st.file_uploader("Upload match video", type=["mp4"])
if uploaded is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.mp4"
        output_path = Path(tmpdir) / "annotated.mp4"
        input_path.write_bytes(uploaded.read())

        try:
            with st.spinner("Processing video..."):
                summary = process_video(str(input_path), str(output_path), DEFAULT_CONFIG)
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            st.stop()

        if not output_path.exists():
            st.error("Processing completed but no annotated output video was created.")
            st.stop()

        video_bytes = output_path.read_bytes()

        st.success("Processing complete")
        st.video(video_bytes)
        st.subheader("Player Metrics")
        if summary:
            st.json(summary)
        else:
            st.info("No players were tracked in this clip.")
