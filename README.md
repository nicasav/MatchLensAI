# MatchLens AI

MatchLens AI is a modular computer vision and video analysis starter repository for amateur football (soccer) match footage.

## Project Structure

```text
.
├── app.py
├── main.py
├── requirements.txt
└── src/
    ├── detection/
    ├── tracking/
    ├── homography/
    ├── metrics/
    └── ui/
```

## Features

- **Detection**: Ultralytics YOLO detects people (`class 0`) and sports ball (`class 32` in COCO).
- **Tracking**: YOLO tracking mode (ByteTrack by default) keeps player IDs consistent across frames.
- **Homography**: OpenCV perspective transform projects player points to a top-down pitch map.
- **Metrics**: Basic per-player estimates for distance covered and average speed, with placeholders for heatmaps/pass tracking.
- **UI**: Minimal Streamlit app to upload and process `.mp4` files.

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Running Video Processing (CLI)

```bash
python main.py --video /path/to/match.mp4 --output /path/to/annotated.mp4
```

Optional config file usage:

```bash
python main.py --video /path/to/match.mp4 --config /path/to/config.json
```

Example config structure:

```json
{
  "model_path": "yolov8n.pt",
  "tracker": "bytetrack.yaml",
  "confidence": 0.25,
  "fps": 30.0,
  "homography": {
    "src_points": [[100, 200], [900, 200], [1000, 700], [80, 700]],
    "dst_points": [[0, 0], [105, 0], [105, 68], [0, 68]]
  }
}
```

Homography note: `src_points` and `dst_points` must be corresponding corners in the same order (for example, clockwise from top-left) for `cv2.getPerspectiveTransform` to produce a correct projection.

## Running the Streamlit App

```bash
streamlit run app.py
```

Then upload an `.mp4` match clip to view:
- annotated output video
- per-player tracking metrics summary

## Public Roadmap

- Add team-color classification and tactical role clustering.
- Improve homography calibration UI with pitch keypoint selection.
- Add event extraction (passes, shots, recoveries).
- Extend metrics engine with possession zones and sprint detection.
- Add persistent storage and export (CSV/JSON/dashboard APIs).
