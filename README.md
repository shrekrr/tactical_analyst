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
2D Coordinate Transformation (metres)
    │
    ▼
Analytics Engine
├── Player metrics  (distance, speed, heatmap, zones)
├── Team metrics    (width, depth, compactness, defensive line)
├── Formation       (rule-based K-Means → Temporal Transformer)
├── Possession      (nearest-player + rolling-window smoothing)
└── Tactical phases (positional heuristics)
    │
    ▼
FastAPI REST + WebSocket backend
    │
    ▼
React + TypeScript + Tailwind dashboard
```

---

## Dataset Sources

### SoccerNet Tracking
Primary dataset for player detection, tracking, and position data.

- Register free at https://www.soccer-net.org/
- Annotations: `Labels-GameState.json` per sequence (bounding boxes + calibration + track IDs)
- Camera calibration included per sequence
- Used for: player positions → sequence building → model training

### SportsMOT _(planned extension)_
Additional football tracking dataset for robustness. Currently not required for MVP.

> **No proprietary broadcast footage is required or redistributed.**
> Training uses open academic datasets only.

---

## Dataset Preparation

### Quick Start (mock data — no account needed)
```bash
# Step 1: generate synthetic data that matches SoccerNet format
python scripts/download_soccernet.py --mock

# Step 2: convert to unified JSONL format
python scripts/prepare_tracking_data.py

# Step 3: build sliding-window sequences + weakly supervised labels
python scripts/create_sequences.py --window 20 --stride 5

# Step 4: inspect class balance and verify no leakage
python scripts/inspect_dataset.py
```

### Real SoccerNet Data
```bash
# Register at https://www.soccer-net.org/ → get a password
export SOCCERNET_PASSWORD="your_password_here"

# Download dev subset (5 sequences, ~500 MB)
python scripts/download_soccernet.py --subset dev

# Then prepare + create as above
python scripts/prepare_tracking_data.py
python scripts/create_sequences.py --window 20 --stride 5
```

### Data Split Strategy
Sequences are split **by match** (never by frame) to prevent leakage:

| Split | Fraction | Example |
|---|---|---|
| Train | 80% | match_A, match_B, match_C |
| Val | 10% | match_D |
| Test | 10% | match_E |

The `split_manifest.json` records the full assignment for reproducibility.

### Label Strategy
Formation labels are **weakly supervised** — derived from the rule-based K-Means detector, not human annotators. This is clearly documented in the `split_manifest.json` and displayed in the UI.

---

## Training

### Supported Models

| Model | Architecture | Temporal modeling |
|---|---|---|
| `mlp` | Linear → LayerNorm → GELU × 3 | None (flattened) |
| `lstm` | BiLSTM (2 layers) → concat h_n → MLP | Sequential |
| `transformer` | Pre-LN Transformer Encoder (3L, 4H) → pool → MLP | Attention-based |

### Train a Single Model
```bash
# From project root
PYTHONPATH=backend python training/train.py \
    --model transformer \
    --config training/configs/transformer_config.yaml
```

Replace `transformer` with `mlp` or `lstm` as needed.

### Compare All Three Models
```bash
PYTHONPATH=backend python training/compare_models.py \
    --config training/configs/transformer_config.yaml
```

Outputs a table like:
```
Model          Accuracy    Macro F1      Params    Train(s)
──────────────────────────────────────────────────────────
mlp              72.3%       0.681      52,748         12s
lstm             78.1%       0.733     201,230         34s
transformer      81.4%       0.772     301,194         51s
```

> **Note:** These metrics measure consistency with the rule-based detector (weakly supervised labels), not accuracy against human-annotated ground truth.

### TensorBoard
```bash
tensorboard --logdir models/
```

### Checkpoints
Best model per architecture saved to:
```
models/
├── mlp/best_model.pt
├── lstm/best_model.pt
└── transformer/best_model.pt
```

Once `models/transformer/best_model.pt` exists, the backend automatically uses it instead of the rule-based fallback.

---

## Computer Vision Pipeline

### Detection — YOLOv8
- Auto-detects CUDA at startup
- Swappable via `ObjectDetector` ABC (`backend/app/detection/base.py`)
- YOLOv8n weights auto-downloaded on first run (~6 MB)

### Tracking — ByteTrack
- Persistent IDs across frames via ByteTrack (built into ultralytics `.track()`)
- Handles occlusion and re-entry

### Team Classification
- Crops jersey region (upper 40% of bounding box, skipping head)
- HSV histogram per player → K-Means (k=2) after accumulating samples
- No team colours hardcoded

### Pitch Calibration
- User selects visible pitch landmarks in the first frame (UI)
- OpenCV `findHomography` with RANSAC computes 3×3 perspective transform
- Converts pixel coordinates → pitch-space metres (105m × 68m)

---

## Deep Learning — Temporal Transformer

### Input Format
```
Shape: (batch, T=20, N×F)
  T = window length (frames sampled at 5 FPS = 4 seconds)
  N = 11 players (padded with zeros if fewer detected)
  F = 4 features: normalised_x, normalised_y, velocity_x, velocity_y
```

### Normalization
- `x / 105.0` → [0, 1]
- `y / 68.0` → [0, 1]
- velocities clipped to [-1, 1] (pitch-lengths per frame)

### Architecture
```
(batch, T, N×F)
      ↓
Linear Projection → d_model=128
      ↓
Sinusoidal Positional Encoding
      ↓
Transformer Encoder (3 layers, 4 heads, Pre-LN)
      ↓
Global Average Pooling over T
      ↓
MLP Head (128 → 64 → n_classes)
      ↓
Formation logits (9 classes)
```

---

## Evaluation

Run evaluation on a trained model:
```bash
PYTHONPATH=backend python training/evaluate.py \
    --model-path models/transformer/best_model.pt \
    --model-type transformer
```

Output:
- `models/transformer/metrics.json` — accuracy, macro F1, precision, recall
- `models/transformer/confusion_matrix.png` — per-class confusion matrix

---

## Installation

### Prerequisites
- Python 3.10+
- Node.js 20+
- NVIDIA GPU recommended (CUDA auto-detected)

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

# Start API server
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## Running Tests
```bash
# From project root
PYTHONPATH=backend python -m pytest tests/ -v

# Output: 58 passed
```

Tests cover:
- Distance, speed, zone analytics
- Homography transforms
- Rule-based formation detection
- Model forward passes (MLP, LSTM, Transformer)
- Sequence building, split integrity, augmentation

---

## User Flow

1. Open `http://localhost:5173` → **Dashboard**
2. Click **Upload Match** → drag a football video (MP4/MOV/AVI/MKV)
3. Select analysis FPS (5 recommended) → **Upload & Analyze**
4. Watch the progress bar (detection → tracking → calibration → analytics)
5. Click **View Analysis** → split view: video + 2D pitch
6. Click any player dot → Player detail (speed chart, heatmap, zone %)
7. Click timeline entries → seek video to formation change

> **Pitch calibration**: For metre-accurate coordinates, submit `POST /api/matches/{id}/calibrate` with 4+ image→pitch point correspondences after uploading.

---

## GPU Support

CUDA auto-detected at startup. No configuration needed:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Minimum VRAM for YOLOv8n: 4 GB.
Tested on: RTX 4050 laptop GPU.

---

## Limitations

1. Speed estimates derived from video pixel movement + homography — not GPS-grade.
2. Pitch calibration requires manual landmark selection.
3. Team classification degrades when jersey colours are similar.
4. Formation labels are weakly supervised (rule-based, not human-annotated).
5. No real-time processing — offline clips only.
6. Ball detection unreliable in low-resolution or blurry footage.
7. Tracking ID switches may occur during heavy occlusion.

---

## Repository Structure

```
tactical_analyst/
├── frontend/              React + TypeScript + Vite + Tailwind UI
├── backend/
│   └── app/
│       ├── api/           FastAPI endpoints
│       ├── detection/     YOLOv8 detector (swappable ABC)
│       ├── tracking/      ByteTrack wrapper
│       ├── teams/         K-Means team classifier
│       ├── pitch/         Calibration + homography
│       ├── analytics/     Player + team metrics + formation
│       ├── models/        MLP, LSTM, Temporal Transformer
│       └── processing/    Pipeline orchestrator
├── training/
│   ├── preprocessing/     Dataset + augmentation
│   ├── configs/           YAML hyperparameter files
│   ├── train.py           Unified training factory
│   ├── compare_models.py  MLP vs LSTM vs Transformer
│   └── evaluate.py        Confusion matrix + metrics
├── scripts/
│   ├── download_soccernet.py     Download or mock data
│   ├── prepare_tracking_data.py  MOT → JSONL
│   ├── create_sequences.py       Build training arrays
│   └── inspect_dataset.py        Class balance + split check
├── tests/                 58 unit tests (pytest)
├── data/                  raw/ processed/ (gitignored)
├── models/                Trained weights (gitignored)
└── docker-compose.yml
```

---

## Future Work

- Automatic pitch line detection (no manual calibration)
- Jersey number recognition
- Pass / shot / carry event detection
- Expected Threat (xT) computation
- Live analysis pipeline
- Natural-language tactical querying (LLM + computed stats only)
- Player identity recognition across matches
- Match-to-match tactical comparison
- SportsMOT integration for tracking robustness

---

## License

MIT — see LICENSE file.

> For educational and research purposes. Not affiliated with the Premier League or any official broadcast organisation.
