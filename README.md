# hubert-mamba

HuBERT-Mamba is a masked unit pretraining implementation for exploring
Mamba-based speech self-supervised models. The repository currently focuses on
HuBERT-style pretraining only: deepfake, ASVspoof, EER evaluation, and LineBot
components have been removed.

## Highlights

- HuBERT-style waveform frontend with a 7-layer convolution feature extractor.
- Mamba encoder variants: `mamba`, `extbimamba`, `innbimamba`, and `mamba_mlp`.
- Masked unit prediction with fairseq-style manifests and `.km` unit labels.
- Token-based batching for variable-length speech samples.
- Optional `mamba_ssm` integration with a local PyTorch fallback for development.
- Training, validation, checkpointing, resume, and optional Weights & Biases logging.

## Project Structure

```text
hubert-mamba/
  configs/
    hubert_mamba/
      mamba_base_iter1.yaml      # HuBERT-Mamba base iter1 preset
      mamba_base_iter2.yaml      # HuBERT-Mamba base iter2 preset
  reports/
    mamba_hubert_change_report.md
  src/
    main.py                     # CLI entrypoint
    config/                     # dataclass configs and YAML loader
    controller/                 # train/eval loop and checkpoint handling
    data/                       # manifest dataset, collate, token sampler
    model/
      hubert_mamba/             # HuBERT-Mamba model and loader
    utils/                      # logging, seeds, TF32, freeze helpers
  pyproject.toml                # package metadata and dependencies
  uv.lock                       # locked dependency graph
```

## Requirements

- Python 3.12
- `uv` for environment and lockfile management
- PyTorch 2.8.0 and torchaudio 2.8.0
- CUDA is recommended for training. CPU is useful for small smoke tests only.

The project does not install `mamba-ssm` by default. If it is available, the
model uses the official Mamba implementation. If it is not available, the model
falls back to `TorchMambaFallback`, which is intended for local validation and
small development runs.

For paper-faithful training, install `mamba-ssm` and its CUDA dependencies in a
compatible Linux/CUDA training environment.

## Setup

```bash
git clone https://github.com/kevinchien920819/hubert-mamba.git
cd hubert-mamba
uv sync
```

Run commands from the repository root. Use `PYTHONPATH=src` so the source tree is
importable without packaging the project first.

```bash
PYTHONPATH=src uv run python src/main.py --config-name=hubert_mamba/mamba_base_iter1
```

## Data Format

The dataset loader expects the fairseq HuBERT layout:

```text
manifest_dir/
  train.tsv
  valid.tsv
label_dir/
  train.km
  valid.km
```

Each manifest starts with one root directory line. Every following line contains
an audio path and the number of waveform samples, separated by a tab.

```text
/path/to/LibriSpeech
train-clean-100/19/198/19-198-0000.flac    225360
train-clean-100/19/198/19-198-0001.flac    182400
```

Audio paths may be relative to the manifest root or absolute paths. Each `.km`
label file must contain one whitespace-separated unit-id sequence per utterance,
aligned with the manifest order.

```text
12 44 44 7 91 3
8 8 19 20 21
```

Key data config fields:

- `data.manifest_dir`: directory containing `<split>.tsv`
- `data.label_dir`: directory containing `<split>.<label>`
- `data.labels`: label suffix list; the loader currently uses the first entry
- `data.label_rate`: unit label frame rate
- `data.num_classes`: number of discrete target units
- `data.max_sample_size`: crop length for long waveforms
- `data.min_sample_size`: minimum length when `pad_audio` is false

## Configuration

Preset configs live in `configs/hubert_mamba/`.

| Config | Purpose | Unit Labels | Updates |
| --- | --- | --- | --- |
| `hubert_mamba/mamba_base_iter1` | First HuBERT-Mamba base pretraining pass | MFCC/k-means style labels | 250k |
| `hubert_mamba/mamba_base_iter2` | Second pass using iter1-derived labels | HuBERT-Mamba iter1 labels | 400k |

Before training, copy or edit a YAML preset and update at least:

```yaml
data:
  manifest_dir: /path/to/LibriSpeech/manifest
  label_dir: /path/to/LibriSpeech/feature/mfcc
  num_classes: 100

general:
  device: cuda
  device_id: '0'

wandb:
  enable: false
```

When `general.work_dir` is `default`, outputs are written to:

```text
outputs/hubert_mamba/<model.name>/<model.tag>/
```

This directory stores log files and checkpoints such as `checkpoint_last.pt` and
`checkpoint_best.pt`.

## Train

Run iter1:

```bash
PYTHONPATH=src uv run python src/main.py --config-name=hubert_mamba/mamba_base_iter1
```

Run iter2 after preparing labels from the iter1 run:

```bash
PYTHONPATH=src uv run python src/main.py --config-name=hubert_mamba/mamba_base_iter2
```

The training loop performs periodic validation and checkpoint saves according to:

- `solver.log_interval_updates`
- `solver.validate_interval_updates`
- `solver.save_interval_updates`
- `solver.update_freq`

## Evaluate

Create a validation-only YAML by setting:

```yaml
general:
  train: false
  eval: true
  testing_ckpt: /path/to/checkpoint_best.pt
```

Then run the same entrypoint:

```bash
PYTHONPATH=src uv run python src/main.py --config-name=hubert_mamba/my_eval_config
```

If `general.ckpt.path` is empty in eval-only mode, the entrypoint loads
`general.testing_ckpt`.

## Resume Training

To resume from the latest checkpoint under the current `work_dir`, set:

```yaml
general:
  ckpt:
    path: self
```

To resume from an explicit checkpoint:

```yaml
general:
  ckpt:
    path: /path/to/checkpoint_last.pt
```

The controller restores model weights, optimizer state, scheduler state, update
count, best validation loss, and AMP scaler state when those fields exist in the
checkpoint.

## Development Checks

Useful local checks:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/hubert-mamba-pycache python3 -m compileall -q src
PYTHONPATH=src uv run python -c "from config import load_config; load_config('hubert_mamba/mamba_base_iter1'); load_config('hubert_mamba/mamba_base_iter2')"
uv lock --check
```

## Notes

- The main CLI only accepts `--config-name`; change runtime options in YAML.
- Preset configs disable WandB by default. Set `wandb.enable: true` to log runs.
- Large datasets, checkpoints, generated labels, and experiment outputs should
  stay outside git.
- See `reports/mamba_hubert_change_report.md` for implementation notes and the
  current validation history.
