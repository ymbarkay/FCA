# FCA — Feedback-based Continual Adaptation

Real-time human-in-the-loop adaptation for the SunFounder PiCar-V using an
EdgeTPU feature extractor and a trainable PyTorch policy head on the CPU.

Legacy scalar/base-model modes are still available for compatibility.

## What this does

The car drives autonomously and can be corrected live by a human operator.
The current deep-mode pipeline is:

- Feature extractor (`.tflite`) runs on the Edge TPU
- PyTorch policy head runs on the CPU and is updated online during teaching

When the human spots bad behaviour, they can:

1. Press **Q** to enter **REVERSE_MANUAL** mode
2. Hold **B** to reverse while steering manually with **A/D/S**
3. Press **R** to resume autopilot or **T** to enter **TEACH**
4. In TEACH mode, commit labels using **W / Shift+W / X**
5. Each commit immediately updates the CPU policy head

The TFLite feature extractor is never modified. Only the PyTorch policy head is.

The current online-learning path combines mixed teach updates, multi-timescale
rehearsal memory, EWC plus fixed-anchor consolidation, historical-gradient
stabilization, and validated manual saves. A manual save marks a known-good
state: it reinforces retention, refreshes the fixed anchor, snapshots a frozen
validated exemplar bank for later rehearsal, and prevents background autosaves
from silently overwriting that checkpoint.

## Project structure

```text
fca/
├── run.py                       Entry point
├── requirements.txt             Pi dependencies
├── requirements-analysis.txt    Extra analysis-studio dependencies
├── README.md                    This file
├── LICENSE                      MIT license
├── checkpoints/                 Adapter weights (auto-saved)
├── logs/                        Per-session CSVs + dataset captures
├── tflite_models/               Place your .tflite files here
├── train_live_policy_head.py    Offline trainer for the live policy head
├── fca_analysis_studio/         Streamlit analysis side app
└── fca/
    ├── core/
    │   ├── state.py             Shared mode + telemetry
    │   ├── teach_state.py       Selected angle controller
    │   └── session_logger.py    Per-session CSVs + dataset export
    ├── perception/
    │   ├── base_model.py        Legacy INT8 TFLite wrapper (scalar/none modes)
    │   └── feature_extractor.py EdgeTPU/CPU feature extractor wrapper
    ├── learning/
    │   ├── adapter_scalar.py    Legacy scalar residual adapter
    │   ├── live_policy_head.py  Trainable PyTorch policy heads + compatibility loader
    │   ├── controller.py        Inference + online update orchestration
    │   ├── paradigms/           Registry-driven dense/MoE online learning variants
    │   ├── replay_buffer.py     Replay buffer helper
    │   └── trainer.py           Optional background gradient step thread
    ├── control/
    │   ├── motors.py            PiCar wrapper
    │   └── driving_loop.py      Main control thread
    └── gui/
        ├── server.py            Flask + WebSocket server
        └── templates/
            └── index.html       Browser UI
```

## Setup on the Raspberry Pi

### 1. Stop SunFounder's auto-server

```bash
sudo systemctl stop picar-server.service
sudo systemctl disable picar-server.service
```

### 2. Free disk space

```bash
df -h /
sudo rm -rf /tmp/pip-unpack-* /tmp/pip-*
sudo journalctl --vacuum-size=100M
sudo apt clean && sudo apt autoremove -y
pip cache purge
df -h /
```

### 3. Install dependencies

```bash
sudo apt install tmux
tmux
pip install --upgrade pip
pip install -r requirements.txt
# If pip runs out of memory: pip install -r requirements.txt --no-cache-dir
```

Optional for the Streamlit analysis side app:

```bash
pip install -r requirements.txt -r requirements-analysis.txt
```

### 4. Place your frozen models

```bash
mkdir -p tflite_models
cp /path/to/feature_extractor_dense512_int8_edgetpu.tflite tflite_models/
```

Optional for legacy scalar/base-only modes:

```bash
cp /path/to/best_model_finetuned_int8_edgetpu.tflite tflite_models/
```

Optional for frozen Keras autopilot models:

```bash
cp /path/to/frozen_driver.keras checkpoints/
# or
cp /path/to/frozen_driver.h5 checkpoints/
```

### 5. Verify imports work

```bash
python3 -c "import tflite_runtime.interpreter; print('tflite OK')"
python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import picar; picar.setup(); print('picar OK')"
```

If you want to run a frozen `.keras` / `.h5` model, TensorFlow must also be
installed in that environment.

### 6. Run

```bash
# First test — no motors, dry run
python3 run.py --mode test --adapter deep \
  --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite \
  --checkpoint checkpoints/live_policy_head_512.pt \
  --learning-paradigm moe_v6_1_stabilized_ownership \
  --learning-rate 1e-4 \
  --no-trainer

# Current deep mode (TPU feature extractor + CPU PyTorch policy head)
python3 run.py --mode drive --adapter deep \
  --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite \
  --checkpoint checkpoints/live_policy_head_512.pt \
  --learning-paradigm moe_v6_1_stabilized_ownership \
  --learning-rate 1e-4 \
  --no-trainer

# Real driving, base model only (no adapter, no learning)
python3 run.py --mode drive --adapter none

# Frozen Keras/H5 autopilot (no adapter, no online learning)
python3 run.py --mode drive --adapter none \
  --base-model checkpoints/frozen_driver.keras

python3 run.py --mode drive --adapter none \
  --base-model checkpoints/frozen_driver.h5

# Legacy scalar residual adapter mode (requires base model file)
python3 run.py --mode drive --adapter scalar \
  --base-model tflite_models/best_model_finetuned_int8_edgetpu.tflite

# Headless mode (no GUI server)
python3 run.py --mode drive --adapter deep --no-gui \
  --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite
```

`--no-trainer` disables the optional background replay-training thread. Teach
commits (`W` / `Shift+W` / `X`) still perform immediate event-triggered updates.

Deep mode also supports registry-driven learning paradigms through
`--learning-paradigm`. Current options include the dense single-head baseline,
Gate-Balanced MoE, Intent-Routed MoE, Intent-Supervised Plastic Experts,
Stabilized Ownership MoE, and Turn-Sharpened Plasticity MoE.

Open `http://<pi-ip>:5000` from your laptop browser.

The GUI now includes a `Max auto speed` control that changes the runtime
autopilot throttle limit without restarting the process.

The GUI also includes an `Inference` selector that lets you switch AUTOPILOT
between the main online model and a frozen model path at runtime. To switch
back to the main online model, start the process with `--adapter deep` or
`--adapter scalar`; if you start with `--adapter none`, only frozen-model
inference is available.

In online runs, the GUI `Adapter` section also lets you switch the active
policy-head checkpoint by choosing from the available `.pt` files in the
checkpoint directory. The newly selected head becomes the active online model
and subsequent saves continue to write to that file.

When `--adapter deep` is active, the GUI also exposes the available online
learning paradigms discovered from `fca/learning/paradigms/`, so you can swap
between dense and MoE variants without editing code.

## Analysis studio

The repo now includes `fca_analysis_studio/`, a separate Streamlit app for:

- Feature probes and extractor comparison
- Feature-space comparison for paper-ready PCA/UMAP figures
- Expert usage / MoE routing analysis
- Transfer-metric summaries
- Latency summaries and export tables

Run it directly as its own standalone app:

```bash
streamlit run fca_analysis_studio/app.py --server.address 0.0.0.0 --server.port 8501
```

If the browser loads the Streamlit shell but then reports that the server is not responding, inspect `logs/analysis_studio.log` on the Pi. That usually means the app failed during startup or import, not that the Pi needs internet access.

The studio writes generated PNG/CSV/TEX artifacts under
`fca_analysis_studio/outputs/`; these are derived outputs rather than source
files.

## Usage workflow

### First-time validation

1. Stop SunFounder's server.
2. Run `python3 run.py --mode drive --adapter none`.
3. Open the GUI and switch to AUTOPILOT.
4. Confirm the car drives correctly on the trained track.
5. Press SPACE and confirm the car stops.
6. Press Q and confirm REVERSE_MANUAL appears.
7. Hold B and confirm the car reverses while A/D steer.
8. Press R and confirm the car returns to AUTOPILOT.

### Collecting teaching data (deep mode)

1. Run with `--adapter deep --feature-model ... --checkpoint ...`.
2. Drive a new track or changed environment.
3. When the model fails, press Q to enter REVERSE_MANUAL.
4. Hold B and use A/D/S to reposition manually.
5. Press T to enter TEACH.
6. Use A/D/S to select the correct angle.
7. Press W / Shift+W / X to commit labels and update the policy head.
8. Press R to resume autopilot.
9. Repeat as needed.
10. Press `P` or click **Save** once behaviour is validated to lock in the
    checkpoint and refresh the validated exemplar bank.

### Dataset collection mode

1. Press `4` or click DATASET to enter `DATASET_COLLECTION`.
2. The car moves forward continuously and captures frames with `speed=1.0`.
3. Press `Z` to stop dataset motion; this also captures one stop-labeled frame.
4. Press `C` to capture a one-shot stop frame from any mode.
5. Press `V` to resume continuous dataset capture.

### Looking at the data

Each session writes outputs under `logs/`:

- `<timestamp>_<session>_frames.csv` — per-frame state snapshots
- `<timestamp>_<session>_corrections.csv` — TEACH correction events
- `<timestamp>_<session>_frames/` — optional saved correction-event frames
- `<timestamp>_<session>_dataset/` — dataset-mode PNG frames plus `train.csv`

AUTOPILOT frame CSV logging is off by default to reduce runtime overhead. When you need full evaluation traces, enable `Auto frame logs` in the operator UI Learning panel before the run. Those frame logs include the MoE gate and intent columns when the online deep model is active.

Dataset export format in `<timestamp>_<session>_dataset/train.csv`:

```text
image_id,angle,speed
```

Where:

- `image_id` matches `0.png`, `1.png`, ...
- `angle = (angle_car - 50) / 80`
- `speed` is `1.0` for moving captures and `0.0` for stop captures

## Adapter modes

### `--adapter scalar` (legacy)

Adapter input: `[base_angle_norm, base_speed_prob]`.

Useful for compatibility testing and quick bring-up.

### `--adapter deep` (current)

Inference and online adaptation use 512-d visual features:

- Feature extractor runs on TPU (`feature_extractor_dense512_int8_edgetpu.tflite`)
- A Mixture-of-Experts policy head (v4 intent routing) runs on CPU
  - Shared stem → 4 independent experts mixed by a learned gate
  - Gate is conditioned on a predicted semantic intent (stop/left/straight/right)
  - Loss: angle CE + speed BCE + intent CE + MoE load-balance + gate entropy
- TEACH updates train the head directly; feature extractor is never modified
- The GUI can switch AUTOPILOT between this online path and a frozen model path
  at runtime

### `--adapter none`

Runs frozen/base-only driving with no online learning. Supported frozen model
formats are `.tflite`, `.keras`, and `.h5`.

## Reproducibility

### Stable profile (validated)

The following configuration is the current validated baseline for online driving
experiments. It was selected because it reduced forgetting significantly while
maintaining useful adaptation speed and improved cross-circuit generalization.

Pin these settings when reproducing results:

- Deep runtime path:
  - EdgeTPU feature extractor (`--adapter deep --feature-model ...`)
  - CPU PyTorch MoE policy head updates (`4` experts on top of shared 512-d features)
- Runtime/logging profile:
  - AUTOPILOT frame logging disabled
  - AUTOPILOT anchor insertion disabled
  - Optional background trainer disabled (`--no-trainer`)
- Teach update profile (`fca/control/driving_loop.py`):
  - `BOOST_STEPS_PER_COMMIT = 5`
  - `BOOST_LR_MULTIPLIER = 2.8`
  - `BOOST_TARGET_REPEATS = 4`
  - `BOOST_REHEARSAL_BATCH_SIZE = 14`
  - Boost cap in dynamic scaling: max 10 steps, max LR multiplier 4.5
  - Drift-conditioned consolidation:
    - `LONG_HORIZON_DRIFT_DECAY_START = 7.5e-4`
    - `LONG_HORIZON_DRIFT_DECAY_END = 3.0e-3`
    - `LONG_HORIZON_MIN_BOOST_SCALE = 0.45`
    - `LONG_HORIZON_MAX_REHEARSAL_SCALE = 2.6`
- Anti-forgetting profile (`fca/control/driving_loop.py`):
  - Hybrid rehearsal memory:
    - `REHEARSAL_RECENT_CAPACITY = 256`
    - `REHEARSAL_PROTECTED_CAPACITY = 768`
    - `REHEARSAL_PROTECTED_FRACTION = 0.72`
    - `REHEARSAL_ELDER_CAPACITY = 192`
    - `REHEARSAL_ELDER_FRACTION = 0.24`
    - `REHEARSAL_ELDER_UPDATE_STRIDE = 6`
    - `VALIDATED_EXEMPLAR_CAPACITY = 192`
    - `VALIDATED_EXEMPLAR_FRACTION = 0.18`
  - Rehearsal updates:
    - `REHEARSAL_BATCH_SIZE = 20`
    - `REHEARSAL_STEPS_PER_COMMIT = 4`
    - `REHEARSAL_LR_MULTIPLIER = 0.60`
  - TEACH-only photometric augmentation:
    - `ENABLE_TEACH_PHOTOMETRIC_AUGMENTATION = True`
    - `TEACH_AUGMENT_PROB = 0.35`
- EWC profile (`fca/learning/controller.py`):
  - `EWC_ENABLED = True`
  - `EWC_LAMBDA = 1.5e-3`
  - `EWC_FISHER_DECAY = 0.99`
  - `EWC_WARMUP_STEPS = 8`
  - `FIXED_ANCHOR_ENABLED = True`
  - `FIXED_ANCHOR_LAMBDA = 4e-4`
  - `FIXED_ANCHOR_WARMUP_STEPS = 8`
  - `HISTORICAL_GRADIENT_ENABLED = True`
  - `HISTORICAL_GRADIENT_BLEND = 0.30`
  - `HISTORICAL_GRADIENT_MOMENTUM = 0.90`
  - Validated-save reinforcement:
    - `VALIDATED_FISHER_BOOST = 0.02`
    - `VALIDATED_STABLE_QUANTILE = 0.60`
    - `VALIDATED_FISHER_MAX = 10.0`
    - `VALIDATED_REINFORCEMENT_COUNT_GAIN = 0.15`
    - `VALIDATED_REINFORCEMENT_MAX_MULT = 3.0`
    - `VALIDATED_RETENTION_EWC_GAIN = 0.12`
    - `VALIDATED_RETENTION_ANCHOR_GAIN = 0.16`
    - `VALIDATED_RETENTION_MAX_MULT = 2.5`

Validated-save behavior to preserve during reproduction:

- Use the GUI Save button or `P` only when behaviour is confirmed good.
- Manual save writes the checkpoint atomically and exposes status in the UI.
- Once a manual save is taken, background autosaves are skipped so the
  validated checkpoint is not silently replaced.
- Session rehearsal memories reset on restart, but checkpoint weights and
  `validated_save_count` persist across sessions.

Recommended command template for this profile:

```bash
python3 run.py --mode drive --adapter deep \
  --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite \
  --checkpoint checkpoints/live_policy_head_512.pt \
  --learning-rate 1e-4 \
  --no-trainer
```

Minimal validation protocol:

1. Start from a known checkpoint (`P` to save when stable).
2. Run a baseline lap in AUTOPILOT and record failures.
3. Perform TEACH corrections only at failures.
4. Re-run the same lap and record retained fixes.
5. Run at least one different circuit and note transfer behavior.
6. Save checkpoint and logs.

Acceptance criteria for this profile:

- Corrections remain effective after multiple loops.
- Previously corrected locations do not regress materially.
- New-circuit behavior improves without severe drift.

For paper-bound runs, record and pin the following:

- Python version (`python3 --version`)
- Exact package versions (`pip freeze > requirements-lock.txt`)
- Model file names and checksums (`sha256sum tflite_models/* checkpoints/*`)
- Random seed(s) for every offline training run
- Exact command lines per condition
- Track/circuit IDs and run order

Recommended run manifest template:

```text
timestamp,condition,track_id,seed,command,checkpoint,notes
```

## Experimental conditions

```bash
# Condition 1: Frozen baseline (no online learning)
python3 run.py --mode drive --adapter none \
  --base-model tflite_models/best_model_finetuned_int8_edgetpu.tflite

# Condition 2: Offline DAgger policy head (trained offline, no online updates)
python3 run.py --mode drive --adapter deep \
  --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite \
  --checkpoint checkpoints/offline_dagger.pt \
  --no-trainer

# Condition 3: Online FCA (live teaching updates)
python3 run.py --mode drive --adapter deep \
  --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite \
  --checkpoint checkpoints/initial.pt \
  --learning-rate 1e-4 \
  --no-trainer
```

## Hardware setup

Document this block exactly for each experiment set:

- Vehicle: SunFounder PiCar-V (version/revision)
- SBC: Raspberry Pi model + RAM
- Camera: model + resolution + frame rate
- Accelerator: Coral USB Accelerator model/revision
- OS: Raspberry Pi OS version + kernel
- Power: battery/adapter spec

## Experimental data

Run artifacts are stored under `logs/`.

- `*_frames.csv`: per-frame telemetry and mode state
- `*_corrections.csv`: TEACH commit events only
- `*_frames/`: optional correction-event frame exports
- `*_dataset/`: dataset-mode PNG frames and `train.csv`

Key columns in `*_frames.csv`:

- `mode`: runtime mode (`AUTOPILOT`, `TEACH`, `REVERSE_MANUAL`, ...)
- `selected_angle_car`: human-selected steering angle
- `dataset_angle_car`: dataset-mode steering label
- `dataset_speed_norm`: dataset-mode speed label (`1` moving, `0` stop frame)
- `inference_ms`, `adapter_ms`: timing metrics
- `replay_buffer_size`, `total_updates`: online learning state

## Offline training

Offline training for the CPU head is handled by `train_live_policy_head.py`.
The current trainer includes:

- chunked validation split to reduce temporal leakage
- class CE plus continuous angle regression
- weighted BCE for speed imbalance
- OneCycleLR, gradient clipping, and early stopping
- best-checkpoint saving compatible with runtime loading

Example:

```bash
python3 train_live_policy_head.py \
  --feature-file X_features_512.npy \
  --y-file y_train.npy \
  --out checkpoints/live_policy_head_512.pt \
  --split-mode chunked \
  --chunk-size 64 \
  --use-balanced-sampler
```

## Keyboard reference

```text
ALWAYS
  1            → PAUSED
  2            → AUTOPILOT
  3            → TEACH
  4            → DATASET_COLLECTION
  Space        → emergency PAUSE
  M            → mark interesting event
  Z            → dataset stop motion
  C            → capture one stop frame (any mode)
  V            → dataset resume motion
  0            → reset adapter/head
  P            → save validated checkpoint

AUTOPILOT mode
  Q            → enter REVERSE_MANUAL
  T            → enter TEACH directly

REVERSE_MANUAL mode
  A / D        → adjust selected angle ±5°
  S            → centre selected angle (90°)
  Hold B       → reverse continuously (no learning)
  R            → resume AUTOPILOT

TEACH mode
  A / D        → adjust selected angle ±5°
  S            → centre selected angle (90°)
  W            → forward step (0.4s) — captures frame, learns
  Shift + W    → long forward step (0.7s) — captures frame, learns
  X            → stop label — captures frame as "stop here", learns
  R            → resume AUTOPILOT

DATASET_COLLECTION mode
  continuous capture while moving forward
  Z            → stop motion + capture stop sample
  C            → capture one stop frame with speed label 0
  V            → resume motion + continuous capture
```

## Troubleshooting

**Camera doesn't open** — change `--capture-src 0` to `1` or `2`. Run
`ls /dev/video*` to see available indices.

**`picar.setup()` fails** — enable I2C in `sudo raspi-config` → Interfacing
Options → I2C → Enable, then reboot.

**Browser can't connect** — make sure both your laptop and Pi are on the same
network. Use the Pi's IP, not `raspberrypi.local`.

**Adapter goes haywire** — click "Reset weights" in the GUI to restore the
model head from the configured checkpoint (buffer is kept).

If no checkpoint is available, reset falls back to parameter reinitialization.
For a hard reset, delete the checkpoint file and restart `run.py`.

**Loss isn't decreasing** — check the TEACH correction frequency and the online
telemetry. The current design no longer depends on heavy AUTOPILOT replay.

**Inference latency creeping up** — inspect `feature_ms`, `adapter_ms`,
`loop_ms`, and `other_ms` in the UI to see whether the bottleneck is model-side
or loop-side.

**Car reverses but mostly makes noise** — increase `REVERSE_MANUAL_SPEED` in
`fca/control/driving_loop.py`.

## License

This project is licensed under the MIT License. See `LICENSE`.

