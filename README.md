# hubert-mamba

HuBERT-Mamba masked unit pretraining implementation for the paper direction "An Exploration of Mamba for Speech Self-Supervised Models".

This repo is now focused on HuBERT-Mamba only. Deepfake / ASVspoof trainer, tester, configs, and model code have been removed.

## Train

Update `data.manifest_dir` and `data.label_dir` in the YAML before running:

```bash
PYTHONPATH=src python src/main.py --config-name=hubert_mamba/mamba_base_iter1
PYTHONPATH=src python src/main.py --config-name=hubert_mamba/mamba_base_iter2
```

Expected data layout follows fairseq HuBERT:

```text
manifest_dir/
  train.tsv
  valid.tsv
label_dir/
  train.km
  valid.km
```

Each `.tsv` starts with the audio root directory, followed by tab-separated `relative_or_absolute_audio_path` and `num_samples`. Each `.km` file has one whitespace-separated integer unit sequence per utterance, aligned to the manifest order.

## Mamba Dependency

If `mamba_ssm` is installed, the model uses it. If it is not installed, the model falls back to a local torch mixer so development and smoke tests do not require compiling `causal-conv1d`.

For paper-faithful training, install `mamba-ssm` and its CUDA dependencies in a compatible training environment.
