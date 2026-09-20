# EPL AI Tactical Analyst

> **AI-powered football tactical analysis platform** — detects players, tracks movement, estimates formations, and generates interactive tactical insights directly from match video.

---

## ✅ Project Status

| Component | Status |
|---|---|
| Backend API (FastAPI) | ✅ Complete |
| Frontend Dashboard (React + TypeScript) | ✅ Complete |
| CV Pipeline (YOLOv8 + ByteTrack) | ✅ Complete |
| Analytics Engine (heatmaps, speed, zones) | ✅ Complete |
| Temporal Transformer (formation classifier) | ✅ Trained — `models/transformer/best_model.pt` |
| Data Pipeline (SoccerNet → sequences) | ✅ Complete |
| Unit Tests | ✅ 58 tests passing |

**Pre-trained model exists.** You do not need to retrain to run the application — skip straight to [Quick Start](#quick-start-run-the-app-now).

---

## Quick Start — Run the App Now

The transformer model is already trained. Just start the backend and frontend:

### Option A — Manual (Recommended for Development)

**Terminal 1 — Backend**
```bash
cd backend

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
# python -m venv venv && source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
API is now live at **http://localhost:8000**  
Swagger docs at **http://localhost:8000/docs**

**Terminal 2 — Frontend**
```bash
cd frontend
npm install
npm run dev
```
Dashboard is now at **http://localhost:5173**

---

### Option B — Docker Compose (One Command)

```bash
# GPU support (requires nvidia-container-toolkit)
docker compose up --build

# CPU-only
docker compose up --build
# (GPU is optional — the backend falls back to CPU automatically)
```

Services:
- Backend → http://localhost:8000
- Frontend → http://localhost:5173

---

## Usage — Analysing a Match

1. Open **http://localhost:5173** → click **Upload Match**
2. Drag a football video (MP4 / MOV / AVI / MKV) — up to 2 GB
3. Choose analysis FPS (5 FPS recommended)
4. Click **Upload & Analyze** — watch the progress bar
5. Click **View Analysis** to see:
   - Formation timeline (Transformer inference)
   - Player heatmaps (32 × 21 pitch grid)
   - Speed & distance charts per player
   - Team possession & compactness
   - 2D pitch mini-map (live player positions)

> **Pitch calibration** (optional, for metre-accurate coordinates):  
> After upload, `POST /api/matches/{id}/calibrate` with 4+ image↔pitch point pairs.

---

## Architecture

```
Raw Video
    │
    ▼
Frame Extraction & Sampling (OpenCV, 5 FPS default)
    │
    ▼
Player + Ball Detection (YOLOv8n — auto-downloaded ~6 MB)
    │
    ▼
Multi-Object Tracking (ByteTrack via ultralytics)
    │
    ▼
Team Classification (K-Means on HSV jersey histograms)
    │
    ▼
Pitch Calibration & Homography (OpenCV, RANSAC)
    │
    ▼
2D Coordinate Transformation (metres: 105 × 68 m)
    │
    ▼
Analytics Engine
├── Player metrics   (distance, speed, heatmap 32×21, zones)
├── Team metrics     (width, depth, compactness, defensive line)
├── Formation        (Temporal Transformer → rule-based fallback)
├── Possession       (nearest-player + rolling-window smoothing)
└── Tactical phases  (positional heuristics)
    │
    ▼
FastAPI REST + WebSocket backend  (port 8000)
    │
    ▼
React + TypeScript + Tailwind dashboard  (port 5173)
```

---

## Trained Model

The `TemporalFormationTransformer` has already been trained on SoccerNet `gamestate-2024` data.

| Metric | Value |
|---|---|
| Architecture | Pre-LN Transformer Encoder (3 layers, 4 heads) |
| Parameters | 609,673 |
| Test accuracy | 32.0% (9-class, weakly supervised labels) |
| Macro F1 | 0.176 |
| Best val loss | 1.621 |
| Train time | ~114 seconds |
| Checkpoint | `models/transformer/best_model.pt` (2.4 MB) |

> **Why 32% accuracy?** Formation labels are *weakly supervised* — derived from a rule-based K-Means detector, not human annotators. The model learns consistency with the rule-based detector; real-world tactical accuracy is qualitative. See `models/transformer/confusion_matrix.png` for per-class breakdown.

The backend automatically uses `models/transformer/best_model.pt` when it exists, falling back to the rule-based detector if the file is absent.

---

## Re-Training (Optional)

Only needed if you want to retrain on your own data or experiment with hyperparameters.

### Prerequisites — Data Setup

**Option 1: Mock data (no account needed, instant)**
```bash
python scripts/download_soccernet.py --mock
python scripts/prepare_tracking_data.py
python scripts/create_sequences.py --window 20 --stride 5
python scripts/inspect_dataset.py        # verify class balance
```

**Option 2: Real SoccerNet data**
```bash
# Register free at https://www.soccer-net.org/ to get a password
set SOCCERNET_PASSWORD=your_password_here   # Windows
# export SOCCERNET_PASSWORD=your_password_here  # macOS/Linux

python scripts/download_soccernet.py --subset dev   # ~500 MB, 5 sequences
python scripts/prepare_tracking_data.py
python scripts/create_sequences.py --window 20 --stride 5
python scripts/inspect_dataset.py
```

The SoccerNet password for this project's dataset is stored separately — contact the repository owner.

### Train a Single Model

```bash
# From project root (backend must be on PYTHONPATH)
$env:PYTHONPATH = "backend"   # Windows PowerShell

python training/train.py --model transformer --config training/configs/transformer_config.yaml
python training/train.py --model lstm        --config training/configs/lstm_config.yaml
python training/train.py --model mlp         --config training/configs/mlp_config.yaml
```

On Linux/macOS:
```bash
PYTHONPATH=backend python training/train.py --model transformer --config training/configs/transformer_config.yaml
```

### Train All Three & Compare (Windows PowerShell)

```powershell
# Mock data — fastest way to verify everything works
.\run_training.ps1 -Mode mock -Model all

# Real SoccerNet data
.\run_training.ps1 -Mode real -SoccerNetPassword YOUR_PASSWORD -Model transformer
```

### Evaluate a Saved Model

```bash
PYTHONPATH=backend python training/evaluate.py \
    --model-path models/transformer/best_model.pt \
    --model-type transformer
```

Outputs:
- `models/transformer/metrics.json` — accuracy, macro F1, precision, recall, n_samples
- `models/transformer/confusion_matrix.png` — per-class confusion matrix

### TensorBoard

```bash
tensorboard --logdir models/
# Open http://localhost:6006
```

---

## Model Checkpoints

```
models/
├── transformer/
│   ├── best_model.pt          ← used by backend automatically
│   ├── metrics.json           ← test evaluation results
│   ├── confusion_matrix.png   ← per-class breakdown
│   └── runs/                  ← TensorBoard logs
├── lstm/
│   └── best_model.pt
└── mlp/
    └── best_model.pt
```

---

## Deep Learning — Temporal Transformer

### Input Format
```
Shape: (batch, T=20, N×F)
  T = 20 frames  (sampled at 5 FPS → 4 seconds of play)
  N = 11 players (zero-padded if fewer detected)
  F = 4 features: x_norm, y_norm, vel_x, vel_y
```

### Normalisation
- `x / 105.0` → [0, 1]
- `y / 68.0` → [0, 1]
- Velocities clipped to [−1, 1] (pitch-lengths per frame)

### Architecture
```
(batch, T=20, N×F=44)
      ↓
Linear Projection → d_model=128
      ↓
Sinusoidal Positional Encoding
      ↓
Transformer Encoder (3 layers, 4 heads, Pre-LN, dropout=0.1)
      ↓
Global Average Pooling over T dimension
      ↓
MLP Head: 128 → 64 → n_classes (9 formations)
      ↓
Formation logits → softmax → predicted formation
```

### Supported Formations (9 classes)
`4-4-2`, `4-3-3`, `4-2-3-1`, `4-5-1`, `3-5-2`, `3-4-3`, `5-3-2`, `5-4-1`, `4-1-4-1`

---

## Dataset

- **Source**: SoccerNet `gamestate-2024` (COCO-style annotations, 25 FPS)
- **Coordinate system**: [0, 105] × [0, 68] metres
- **Subsampling**: 25 FPS → 5 FPS
- **Split strategy**: by full match (never by frame — no leakage)

| Split | Fraction |
|---|---|
| Train | 80% |
| Val | 10% |
| Test | 10% |

---

## Installation Details

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| Node.js | 18 | 20 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | — (CPU ok) | 4 GB (CUDA 11.8+) |
| Disk | 5 GB | 15 GB (with SoccerNet) |

### Backend Installation
```bash
cd backend
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

Key dependencies: `fastapi`, `uvicorn`, `ultralytics` (YOLOv8), `torch>=2.2`, `opencv-python-headless`, `scikit-learn`, `SoccerNet`

### Frontend Installation
```bash
cd frontend
npm install
```

Key dependencies: React 18, TypeScript, Vite, Tailwind CSS, Recharts

---

## Running Tests

```bash
# From project root
$env:PYTHONPATH = "backend"   # Windows PowerShell
# PYTHONPATH=backend           # macOS/Linux

python -m pytest tests/ -v
# Expected: 58 passed
```

Tests cover:
- Distance, speed, zone analytics
- Homography transforms
- Rule-based formation detection
- Model forward passes (MLP, LSTM, Transformer)
- Sequence building, split integrity, augmentation

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/matches/upload` | Upload video for analysis |
| `GET` | `/api/matches` | List all analysed matches |
| `GET` | `/api/matches/{id}` | Full match analysis JSON |
| `GET` | `/api/matches/{id}/formations` | Formation timeline |
| `GET` | `/api/matches/{id}/players/{player_id}` | Player metrics |
| `POST` | `/api/matches/{id}/calibrate` | Submit pitch calibration points |
| `WS` | `/ws/matches/{id}/progress` | Real-time processing progress |

Full interactive docs: **http://localhost:8000/docs**

---

## Computer Vision Pipeline Details

### Detection — YOLOv8n
- Weights auto-downloaded on first run (~6 MB via ultralytics)
- CUDA auto-detected — no config required
- Swappable via `ObjectDetector` ABC in `backend/app/detection/base.py`

### Tracking — ByteTrack
- Persistent IDs across frames (built into `ultralytics .track()`)
- Handles occlusion and re-entry

### Team Classification
- Crops jersey region (upper 40% of bounding box, skipping head)
- HSV histogram per player → K-Means (k=2) after accumulating 50+ samples
- No hardcoded team colours

### Pitch Calibration
- `POST /api/matches/{id}/calibrate` with 4+ image↔pitch coordinate pairs
- OpenCV `findHomography` with RANSAC → 3×3 perspective transform
- Converts pixel → pitch metres (105 m × 68 m)

---

## Repository Structure

```
tactical_analyst/
├── frontend/                  React + TypeScript + Vite + Tailwind
│   └── src/
│       ├── components/        Heatmap, PitchMap, SpeedChart, ...
│       ├── pages/             Dashboard, MatchAnalysis, Upload
│       └── types/             TypeScript interfaces (analysis.ts)
│
├── backend/
│   └── app/
│       ├── api/               FastAPI routers (matches, players, ws)
│       ├── detection/         YOLOv8 detector (swappable ABC)
│       ├── tracking/          ByteTrack wrapper
│       ├── teams/             K-Means team classifier
│       ├── pitch/             Calibration + homography
│       ├── analytics/         Player metrics, team metrics, formation
│       ├── models/            MLP, LSTM, Temporal Transformer (PyTorch)
│       ├── processing/        Pipeline orchestrator (pipeline.py)
│       └── schemas.py         Pydantic response models
│
├── training/
│   ├── preprocessing/         Dataset + augmentation (soccernet_loader.py)
│   ├── configs/               YAML hyperparameter files (3 models)
│   ├── train.py               Unified training factory (MLP/LSTM/Transformer)
│   ├── compare_models.py      Train all 3 and print comparison table
│   └── evaluate.py            Confusion matrix + metrics JSON
│
├── scripts/
│   ├── download_soccernet.py  Download real data or generate mock data
│   ├── prepare_tracking_data.py  MOT annotations → JSONL
│   ├── create_sequences.py    Build sliding-window training arrays
│   └── inspect_dataset.py     Class balance + train/val/test split check
│
├── models/
│   ├── transformer/
│   │   ├── best_model.pt      ← Pre-trained (2.4 MB) — used by backend
│   │   ├── metrics.json       ← Test evaluation results
│   │   └── confusion_matrix.png
│   ├── lstm/best_model.pt
│   └── mlp/best_model.pt
│
├── tests/                     58 unit tests (pytest)
├── data/                      raw/ processed/ (gitignored)
├── run_training.ps1           Windows PowerShell training helper
└── docker-compose.yml         Full stack Docker deployment
```

---

## GPU Support

CUDA is auto-detected everywhere — no configuration needed:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

- **Training**: ~114 seconds on CPU for the transformer (6 matches of data)
- **Inference**: Real-time capable on CPU; GPU gives ~3× speedup
- **YOLOv8n**: Minimum 4 GB VRAM; tested on RTX 4050 laptop GPU

---

## Known Limitations

1. Speed estimates from pixel movement + homography — not GPS-grade accuracy
2. Pitch calibration requires manual landmark selection (4+ points)
3. Team classification degrades when jersey colours are similar (e.g., white vs light grey)
4. Formation labels are weakly supervised (rule-based, not human-annotated) — 32% test accuracy reflects label noise, not solely model quality
5. Offline analysis only — no real-time pipeline
6. Ball detection is unreliable in low-resolution or motion-blurred footage
7. Tracking ID switches may occur during heavy occlusion

---

## Future Work

- Automatic pitch line detection (remove manual calibration)
- Jersey number recognition
- Pass / shot / carry event detection
- Expected Threat (xT) computation
- Live analysis pipeline (RTSP stream support)
- Natural-language tactical querying (LLM + computed stats)
- Player identity recognition across matches
- SportsMOT integration for tracking robustness

---

## License

MIT — see LICENSE file.

> For educational and research purposes. Not affiliated with the Premier League or any official broadcast organisation.
