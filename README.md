# EPL AI Tactical Analyst

> **AI-powered football tactical analysis platform** — detects players, tracks movement, estimates formations, and generates interactive tactical insights directly from match video.

---

## Architecture

```
Raw Video
    │
    ▼
Frame Extraction & Sampling (OpenCV)
    │
    ▼
Player + Ball Detection (YOLOv8)
    │
    ▼
Multi-Object Tracking (ByteTrack via ultralytics)
    │
    ▼
Team Classification (K-Means on HSV histograms)
    │
    ▼
Pitch Calibration & Homography (OpenCV)
    │
    ▼
2D Coordinate Transformation
    │
    ▼
Analytics Engine
├── Player metrics (distance, speed, heatmap, zones)
├── Team metrics (width, depth, compactness, defensive line)
├── Formation detection (rule-based K-Means + Temporal Transformer hook)
├── Possession estimation (nearest-player + temporal smoothing)
└── Tactical phase detection (positional heuristics)
    │
    ▼
FastAPI REST + WebSocket backend
    │
    ▼
React + TypeScript + Tailwind frontend
```

---

## Computer Vision Pipeline

### 1. Frame Extraction
OpenCV `VideoCapture` extracts frames at a configurable sample rate (default 5 FPS). This dramatically reduces computation while preserving tactical structure.

### 2. Player Detection — YOLOv8
Uses `ultralytics` YOLOv8. Detects:
- **player** (remapped from COCO `person` class)
- **ball** (COCO `sports ball`)
- **referee** (when using a football-specific model)

The `ObjectDetector` abstract base class allows swapping in any model without changing the pipeline.

### 3. Tracking — ByteTrack
Uses the ByteTrack algorithm built into ultralytics `.track()`. Assigns persistent integer IDs across frames. Handles:
- Temporary occlusion
- Players crossing each other
- Camera movement

### 4. Team Classification
For each tracked player:
1. Crops the jersey region (upper 40% of bounding box, skipping the head)
2. Converts to HSV colour space
3. Computes a normalised 2D H+S histogram
4. After accumulating samples, runs K-Means (k=2) to cluster into two teams
5. Derives a representative hex colour per cluster

No EPL team colours are hardcoded.

### 5. Pitch Calibration & Homography
Uses **semi-automatic calibration**: the user clicks known pitch landmarks (corners, penalty spots, centre circle) on the first video frame. OpenCV `findHomography` with RANSAC computes the 3×3 perspective transform matrix.

This maps image-space pixel coordinates → pitch-space metre coordinates (origin = top-left, 105m × 68m standard pitch).

### 6. Feature Engineering
For each tracking point after homography:
- **Distance**: Euclidean sum of consecutive pitch positions
- **Speed**: distance / time, smoothed with Savitzky–Golay filter, converted to km/h
- **Heatmap**: 2D histogram on a 26×17 grid (≈4m cells)
- **Zone time**: Defensive / Middle / Attacking third fractions

Team-level:
- **Width**: max(y) − min(y)
- **Depth**: max(x) − min(x)
- **Centroid**: mean position of outfield players
- **Compactness**: mean distance from centroid
- **Defensive line**: mean X of the 4 deepest players

---

## Deep Learning — Temporal Transformer

### Architecture
```
Player Coordinates (T, N×F)
        │
Linear Feature Projection → d_model=128
        │
Sinusoidal Positional Encoding
        │
Transformer Encoder (3 layers, 4 heads, Pre-LN)
        │
Global Average Pooling over T
        │
MLP Classification Head
        │
Formation label + softmax confidence
```

**Input**: `(batch, T=20, N×F)` where N=11 players, F=4 (x, y, vx, vy)  
**Output**: formation class logit (8 classes)

The model requires training data (sequences of player coordinates labeled with ground-truth formations). Without trained weights, the system falls back to the rule-based formation detector.

### Rule-Based Baseline
1. Collect outfield player X coordinates
2. K-Means cluster into 3–4 rows
3. Count players per row (sorted defensive → attacking)
4. Map count string → formation label
5. Return closest known formation by edit distance

### Training
```bash
cd epl-tactical-analyst
python training/train.py --config training/configs/transformer_config.yaml
```

Training data must be placed in `training/datasets/`:
- `sequences.npy` — shape `(N, T, n_players × n_features)`
- `labels.npy` — shape `(N,)` integer class indices

See `training/configs/transformer_config.yaml` for all hyperparameters.

---

## Evaluation Metrics

| Component | Metrics |
|---|---|
| Detection | mAP, Precision, Recall |
| Tracking | IDF1, MOTA, ID switches |
| Formation (rule-based) | — (deterministic) |
| Formation (Transformer) | Accuracy, Macro F1, Confusion Matrix |
| Tactical phases | Accuracy, Macro F1 |
| Speed estimation | Labelled "AI-estimated" — not GPS-grade |

> ⚠️ **Without training data**, the Temporal Transformer cannot be evaluated. The rule-based baseline is used by default. All metrics are computed on held-out test splits — never fabricated.

---

## Installation

### Prerequisites
- Python 3.10+
- Node.js 20+
- CUDA-capable GPU (optional but recommended)

### Backend
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

---

## Running Locally

1. Start backend: `uvicorn app.main:app --reload`
2. Start frontend: `npm run dev`
3. Open http://localhost:5173
4. Click **Upload Match** and select a football video
5. Set sample FPS (5 recommended)
6. Click **Upload & Analyze**
7. Wait for processing to complete (visible progress bar)
8. Click **View Analysis** to open the dashboard

### Pitch Calibration
After uploading, you'll be prompted to calibrate the pitch by clicking known landmarks on the first video frame. The system uses these to compute the homography matrix. Without calibration, tracking is displayed in image-space (pixels) rather than pitch-space (metres).

---

## GPU Requirements
The system auto-detects CUDA. If available, YOLO inference runs on GPU. Without CUDA, it falls back to CPU (significantly slower).

Minimum GPU: 4 GB VRAM for YOLOv8n  
Recommended: 8 GB VRAM for larger models

---

## Limitations

1. **Speed estimates** are derived from video pixel movement and homography — not GPS. Labelled "AI-estimated" throughout the UI.
2. **Pitch calibration** requires manual keypoint selection. Fully automatic pitch-line detection is a future extension.
3. **Team classification** uses colour clustering — accuracy degrades with similar jersey colours or when the camera angle makes jerseys indistinguishable.
4. **Formation detection** at MVP uses rule-based K-Means clustering. A trained Temporal Transformer is architecturally included but requires labeled training data.
5. **No real-time processing** — the system processes uploaded clips offline.
6. **Tracking ID switches** may occur during heavy occlusion.
7. **Ball detection** is often unreliable in low-resolution or blurry footage.

---

## Future Work

- Automatic pitch line detection (no manual calibration)
- Jersey number recognition
- Pass / shot / carry event detection
- Expected Threat (xT) computation
- Pressing intensity maps
- Live analysis pipeline
- Multimodal commentary + video analysis
- Natural-language tactical querying
- Player identity recognition
- Match-to-match tactical comparison

---

## Repository Structure

```
epl-tactical-analyst/
├── frontend/          React + TypeScript + Vite + Tailwind UI
├── backend/
│   └── app/
│       ├── api/       FastAPI endpoints
│       ├── detection/ YOLO detector (swappable)
│       ├── tracking/  ByteTrack wrapper
│       ├── teams/     K-Means team classifier
│       ├── pitch/     Calibration + homography
│       ├── analytics/ Player + team metrics + formation
│       ├── models/    Temporal Transformer
│       └── processing/ Pipeline orchestrator
├── training/          Train + evaluate the Transformer
├── models/            Trained weight files (not committed)
├── data/              Uploaded videos + results (not committed)
├── notebooks/         Exploration notebooks
└── docker-compose.yml
```

---

## License

MIT — see LICENSE file.

> This project is for educational and research purposes. It is not affiliated with the Premier League or any official broadcast organisation.
