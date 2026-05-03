# HEL — Hybrid Edge Learning

Real-time human-in-the-loop adaptation for the SunFounder PiCar-V running an
EdgeTPU feature extractor plus a trainable PyTorch policy head on the CPU.

Legacy scalar/base-model modes are still available for compatibility.

## What this does

The car drives autonomously and can be corrected live by a human operator.
Current deep mode pipeline is:

- Feature extractor (`.tflite`) runs on Edge TPU
- PyTorch policy head runs on CPU and is updated online during teaching

When the human spots bad behaviour, they can:

1. Press **Q** to enter **REVERSE_MANUAL** mode
2. Hold **B** to reverse while steering manually with **A/D/S**
3. Press **R** to resume autopilot or **T** to enter **TEACH**
4. In TEACH mode, commit labels using **W / Shift+W / X**
5. Each commit immediately updates the CPU policy head

The TFLite feature extractor is never modified. Only the PyTorch policy head is.

## Project structure

```
hel/
├── run.py                       Entry point
├── requirements.txt             Pi dependencies
├── README.md                    This file
├── checkpoints/                 Adapter weights (auto-saved)
├── logs/                        Per-session CSVs + correction frames
├── tflite_models/               Place your .tflite files here
└── hel/
    ├── core/
    │   ├── state.py             Shared mode + telemetry
    │   ├── command_buffer.py    Rewind history
    │   ├── teach_state.py       Selected angle controller
    │   └── session_logger.py    Per-session CSVs
    ├── perception/
    │   ├── base_model.py        Legacy INT8 TFLite wrapper (scalar/none modes)
    │   ├── feature_extractor.py EdgeTPU/CPU feature extractor wrapper
    │   └── model_split.py       Feature pipeline prep utilities
    ├── learning/
    │   ├── adapter_scalar.py    Legacy scalar residual adapter
    │   ├── adapter_deep.py      Legacy deep residual adapter
    │   ├── live_policy_head.py  Trainable PyTorch policy head (current deep mode)
    │   ├── controller.py        Inference + online update orchestration
    │   ├── replay_buffer.py     FIFO of teaching samples
    │   └── trainer.py           Background gradient step thread
    ├── control/
    │   ├── motors.py            PiCar wrapper
    │   ├── rewind.py            Legacy inverse playback helper
    │   └── driving_loop.py      Main control thread
    └── gui/
        ├── server.py            Flask + WebSocket
        └── templates/
            └── index.html       Browser UI
```

## Setup on the Raspberry Pi

### 1. Stop SunFounder's auto-server

```bash
sudo systemctl stop picar-server.service
sudo systemctl disable picar-server.service   
```

### 2. Free disk space (you need at least 4 GB free)

```bash
df -h /
sudo rm -rf /tmp/pip-unpack-* /tmp/pip-*
sudo journalctl --vacuum-size=100M
sudo apt clean && sudo apt autoremove -y
pip cache purge
df -h /     # confirm Avail > 4 GB
```

### 3. Install dependencies (use tmux to survive SSH drops)

```bash
sudo apt install tmux
tmux
pip install --upgrade pip
pip install -r requirements.txt
# If pip runs out of memory: pip install -r requirements.txt --no-cache-dir
```

### 4. Place your INT8 TFLite model

```bash
mkdir -p tflite_models
cp /path/to/feature_extractor_dense512_int8_edgetpu.tflite tflite_models/
```

Optional (legacy scalar/base-only modes):

```bash
cp /path/to/best_model_finetuned_int8_edgetpu.tflite tflite_models/
```

### 5. Verify imports work

```bash
python3 -c "import tflite_runtime.interpreter; print('tflite OK')"
python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import picar; picar.setup(); print('picar OK')"
```

### 6. Run

```bash
# First test — no motors, dry run
python3 run.py --mode test --adapter deep \
  --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite \
  --checkpoint checkpoints/live_policy_head_512.pt \
  --learning-rate 1e-4 \
  --no-trainer

# Current deep mode (TPU feature extractor + CPU PyTorch policy head)
python3 run.py --mode drive --adapter deep \
  --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite \
  --checkpoint checkpoints/live_policy_head_512.pt \
  --learning-rate 1e-4 \
  --no-trainer

# Real driving, base model only (no adapter, no learning)
python3 run.py --mode drive --adapter none

# Legacy scalar residual adapter mode (requires base model file)
python3 run.py --mode drive --adapter scalar \
  --base-model tflite_models/best_model_finetuned_int8_edgetpu.tflite

# Headless mode (no GUI server)
python3 run.py --mode drive --adapter deep --no-gui \
  --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite
```

`--no-trainer` disables the optional background replay-training thread. Teach
commits (`W` / `Shift+W` / `X`) still perform immediate event-triggered updates.

Open `http://<pi-ip>:5000` from your laptop browser.

## Usage workflow

### First-time validation (do this before any experiments)

1. Stop SunFounder's server
2. Run `python3 run.py --mode drive --adapter none`
3. Open the GUI, switch to AUTOPILOT
4. Confirm car drives correctly on the trained track
5. Press SPACE — confirm car stops
6. Press Q — confirm REVERSE_MANUAL mode appears
7. Hold B — confirm car reverses while A/D steer
8. Press R — confirm car returns to AUTOPILOT mode

### Collecting teaching data (deep mode)

1. Run with `--adapter deep --feature-model ... --checkpoint ...`
2. Drive a *new* track (different from training)
3. When the model fails, press Q to enter REVERSE_MANUAL
4. Hold B and use A/D/S to reposition manually
5. Press T to enter TEACH
6. Use A/D/S to select the correct angle
7. Press W / Shift+W / X to commit labels and update policy
8. Press R to resume autopilot
9. Repeat as the model fails

### Dataset collection mode

1. Press `4` or click DATASET to enter `DATASET_COLLECTION`
2. Car moves forward continuously and captures frames with `dataset_speed_norm=1`
3. Press `Z` to stop dataset motion
4. Press `C` to capture a single stop frame (`dataset_speed_norm=0`)
5. Press `V` to resume continuous dataset capture at constant speed

### Looking at the data

Each session writes two CSVs to `logs/`:

- `<timestamp>_<session>_frames.csv` — every frame's state
- `<timestamp>_<session>_corrections.csv` — only teaching events (with frame paths)
- `<timestamp>_<session>_frames/` — saved JPEGs at correction events
- `<timestamp>_<session>_dataset/` — dataset-mode captured JPEGs

`frames.csv` includes dataset labels:

- `dataset_angle_car`
- `dataset_speed_norm`

## Stages

### Adapter mode: `--adapter scalar` (legacy)

Adapter input: `[base_angle_norm, base_speed_prob]` (2 scalars).

Useful for quick bring-up and compatibility testing.

### Adapter mode: `--adapter deep` (current)

Inference and online adaptation use 512-d visual features:

- Feature extractor runs on TPU (`feature_extractor_dense512_int8_edgetpu.tflite`)
- PyTorch `LivePolicyHead` runs on CPU
- Teach updates train the head directly using angle-class CE + speed BCE

### Adapter mode: `--adapter none`

Runs frozen/base-only driving with no online learning.

The old residual deep adapter path is retained in source as legacy code.

Core runtime pipeline (modes, GUI, replay buffer, logging) stays shared.

## Reproducibility

### Stable profile (validated)

The following configuration is the current validated baseline for online driving
experiments. It was selected because it reduced forgetting significantly while
maintaining useful adaptation speed and improved cross-circuit generalization.

Pin these settings when reproducing results:

- Deep runtime path:
  - EdgeTPU feature extractor (`--adapter deep --feature-model ...`)
  - CPU PyTorch live policy head updates
- Runtime/logging profile:
  - AUTOPILOT frame logging disabled
  - AUTOPILOT anchor insertion disabled
  - Optional background trainer disabled (`--no-trainer`)
- Teach update profile (`hel/control/driving_loop.py`):
  - `BOOST_STEPS_PER_COMMIT = 5`
  - `BOOST_LR_MULTIPLIER = 2.8`
  - Boost cap in dynamic scaling: max 10 steps, max LR multiplier 4.5
- Anti-forgetting profile (`hel/control/driving_loop.py`):
  - Hybrid rehearsal memory:
    - `REHEARSAL_RECENT_CAPACITY = 256`
    - `REHEARSAL_PROTECTED_CAPACITY = 512`
    - `REHEARSAL_PROTECTED_FRACTION = 0.60`
  - Rehearsal updates:
    - `REHEARSAL_BATCH_SIZE = 16`
    - `REHEARSAL_STEPS_PER_COMMIT = 3`
    - `REHEARSAL_LR_MULTIPLIER = 0.60`
- EWC profile (`hel/learning/controller.py`):
  - `EWC_ENABLED = True`
  - `EWC_LAMBDA = 1.5e-3`
  - `EWC_FISHER_DECAY = 0.99`
  - `EWC_THETA_MOMENTUM = 0.998`
  - `EWC_WARMUP_STEPS = 8`

Recommended command template for this profile:

```bash
python3 run.py --mode drive --adapter deep \
  --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite \
  --checkpoint checkpoints/live_policy_head_512.pt \
  --learning-rate 1e-4 \
  --no-trainer
```

Minimal validation protocol (per run):

1. Start from a known checkpoint (`P` to save when stable).
2. Run baseline lap in AUTOPILOT and record failures.
3. Perform TEACH corrections only at failures.
4. Re-run the same lap and record retained fixes.
5. Run at least one different circuit and note transfer behavior.
6. Save checkpoint and logs (`*_frames.csv`, `*_corrections.csv`, `*_dataset/train.csv`).

Acceptance criteria for this profile:

- Corrections remain effective after multiple loops.
- Previously corrected locations do not regress materially.
- New-circuit behavior improves without severe drift.

For paper-bound runs, record and pin the following:

- Python version (`python3 --version`)
- Exact package versions (`pip freeze > requirements-lock.txt`)
- Model file names and checksums (`sha256sum tflite_models/* checkpoints/*`)
- Random seed(s) for every offline training run
- Exact command lines per condition (see below)
- Track/circuit IDs and run order

Recommended run manifest template (one line per trial):

```
timestamp,condition,track_id,seed,command,checkpoint,notes
```

## Experimental conditions

Example command templates matching the planned study conditions:

```bash
# Condition 1: Frozen baseline (no online learning)
python3 run.py --mode drive --adapter none \
  --base-model tflite_models/best_model_finetuned_int8_edgetpu.tflite

# Condition 2: Offline DAgger policy head (trained offline, no online updates)
python3 run.py --mode drive --adapter deep \
  --feature-model tflite_models/feature_extractor_dense512_int8_edgetpu.tflite \
  --checkpoint checkpoints/offline_dagger.pt \
  --no-trainer

# Condition 3: Online HEL-RAST (live teaching updates)
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
- `*_corrections.csv`: teach commit events only
- `*_frames/`: correction-event JPEGs
- `*_dataset/`: dataset-mode JPEGs

Key columns in `*_frames.csv`:

- `mode`: runtime mode (`AUTOPILOT`, `TEACH`, `REVERSE_MANUAL`, ...)
- `selected_angle_car`: human-selected steering angle
- `dataset_angle_car`: dataset-mode steering label
- `dataset_speed_norm`: dataset-mode speed label (`1` moving, `0` stop frame)
- `inference_ms`, `adapter_ms`: timing metrics
- `replay_buffer_size`, `total_updates`: online learning state

## Keyboard reference

```
ALWAYS
  1            → PAUSED
  2            → AUTOPILOT
  3            → TEACH
  4            → DATASET_COLLECTION
  Space        → emergency PAUSE
  M            → mark interesting event
  Z            → dataset stop motion
  C            → capture one stop frame (dataset mode)
  V            → dataset resume motion
  0            → reset adapter/head
  P            → save checkpoint

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
  Z            → stop motion
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

For deep mode checkpoints saved from offline training, pass:
`--checkpoint checkpoints/live_policy_head_512.pt`

**Loss isn't decreasing** — check the buffer size in telemetry. Trainer needs
≥ 8 samples to start. Each W/X commit adds one sample.

**Inference latency creeping up** — that's a sign the trainer is competing
with the inference thread. Increase `--update-period` (e.g. from 1.5s to 3s).
This is your RQ3 variable for the paper.

**Car reverses but mostly makes noise** — increase `REVERSE_MANUAL_SPEED` in
`hel/control/driving_loop.py`.

**Want continuous correction (not stepwise)** — the current design is stepwise
only. To add continuous, modify `_handle_teach()` in `driving_loop.py` to also
react to held-down arrow keys and continuously add samples to the buffer.
