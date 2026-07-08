# Mamba-based HuBERT Change Report

## Summary

本次實作已收斂為 HuBERT-Mamba only。依最新需求，所有 deepfake training pipeline、ASVspoof dataset path、EER trainer/tester、LineBot 通知與 deepfake configs/model files 已移除；repo 目前只保留 Mamba-based HuBERT masked unit pretraining。

公開命令維持：

```bash
PYTHONPATH=src python src/main.py --config-name=hubert_mamba/mamba_base_iter1
PYTHONPATH=src python src/main.py --config-name=hubert_mamba/mamba_base_iter2
```

## Current Architecture

- `src/main.py`：HuBERT-Mamba 專用入口，只載入 `HubertMambaConfig`。
- `src/controller/hubert_mamba.py`：單一 `HubertMambaController`，constructor 依 `cfg.general.train/eval` 決定 `train` 或 `eval` mode；不再區分 trainer/tester。
- `src/data/dataset.py`：只保留 fairseq-style manifest + `.km` label 的 `HubertPretrainDataset`。
- `src/data/loader.py`：只保留 `get_hubert_dataloader()`。
- `src/model/hubert_mamba/model.py`：7-layer CNN frontend、Mamba variants、masking、masked unit prediction output contract。
- `src/model/hubert_mamba/model.py`：若沒有 `mamba_ssm`，會使用 dependency-free `TorchMambaFallback`，避免本機編譯 `causal-conv1d`。
- `configs/hubert_mamba/mamba_base_iter1.yaml`、`configs/hubert_mamba/mamba_base_iter2.yaml`：HuBERT-Mamba iter1/iter2 presets。

## Removed Deepfake Code

已刪除：

- `configs/deepfake/*`
- `src/config/deepfake/*`
- `src/model/deepfake/*`
- `src/controller/base.py`
- `src/controller/trainer.py`
- `src/controller/tester.py`
- `src/controller/eval.py`
- `src/tools/linebot.py`

`src/config/base.py` 也已精簡為 HuBERT 共用的 `GeneralConfig`、`DataloaderConfig`、`WandbConfig`，並移除舊 evaluation file 欄位。

`README.md` 已補 HuBERT-Mamba 使用方式、資料格式與 fallback dependency 說明。

## Multi-Agent Work

- Bernoulli：逐項 audit 原 18 個 findings，指出剩餘缺口為 sampler `__len__`、HuBERT pad length、`uv.lock`、deepfake duration。
- Peirce：檢查 `uv.lock` / `causal-conv1d` 狀態，結論是不應在目前 macOS arm64 環境編譯或手改 lockfile；後續已移除 direct CUDA extension dependency。
- Godel：原先處理 deepfake duration；需求改為 HuBERT-only 後已中止，相關 deepfake 改動已隨清理移除。

## 18 Findings Resolution

1. `align_targets_to_length` 越界：已修。
2. `TokenBatchSampler.__len__` 低估：已修為依下一次 iteration 順序 dry-run；無 seed shuffle 回傳保守上界。
3. sampler 每輪同序：已修，seed 加 iteration counter。
4. token batch cost 未反映 padding：已修為 `batch_size * max_len`。
5. sampler 長度未反映 crop：已修，HuBERT dataset `get_lengths()` 使用 `max_sample_size` 上限。
6. `pad_audio=true` 後 padded length 被當有效 supervision：已修，collate 使用 `Sample.length`。
7. 空 manifest / empty dataloader：已修，dataset/controller 直接報錯。
8. 短樣本造成 CNN runtime error：已修，不 pad 時依 `min_sample_size` 過濾。
9. `mamba_mlp` no-op：已修，未設定 FFN 時自動用 `4 * encoder_embed_dim`。
10. bi-Mamba padding 污染：已修，反向 pass 逐樣本只處理 valid frames。
11. `extractor_mode=layer_norm` 未實作：已修。
12. zero-loss backward 失敗：已修，zero loss 保留 graph。
13. checkpoint resume 不恢復 optimizer/scheduler/update/scaler：已修。
14. eval-only 未使用 `testing_ckpt`：已修。
15. top-level model API 空白：已修，`model` lazy export HuBERT classes。
16. top-level optional dependency 脆弱：deepfake dependency 已刪除；HuBERT waveform loading 仍 lazy import `torchaudio`。
17. `uv.lock` 未同步 `mamba-ssm` / `causal-conv1d`：已移除 `pyproject.toml` direct dependencies，避免 locked install 或 macOS 開發環境自動觸發 CUDA extension build；模型在缺 `mamba_ssm` 時使用 local torch fallback。`uv.lock` 也沒有這兩個 package 的 stale package entry。
18. deepfake duration 未實作：不再適用。deepfake training code 已刪除，repo focus 改為 HuBERT-Mamba。

## Dependency Note

`pyproject.toml` direct dependencies 目前只保留一般 Python/PyTorch runtime，不再直接宣告 `mamba-ssm` 或 `causal-conv1d`。

模型載入策略：

- 若環境有 `mamba_ssm`，優先使用官方 Mamba class。
- 若環境沒有 `mamba_ssm`，使用 `TorchMambaFallback`，讓 HuBERT data/model/controller 可在不編譯 `causal-conv1d` 的環境執行 smoke test 或小規模驗證。

目前不在 macOS arm64 本機編譯 `causal-conv1d`，也不手動偽造 `uv.lock`。真實 paper-faithful CUDA Mamba kernel 需在相容訓練環境另外安裝。

## Verification

已執行：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/hubert-mamba-pycache python3 -m compileall -q src
PYTHONPATH=src python3 - <<'PY'  # HuBERT iter1/iter2 config + public import smoke
PYTHONPATH=src python3 - <<'PY'  # TokenBatchSampler len smoke
PYTHONPATH=src python3 - <<'PY'  # HuBERT pad_audio length smoke with fake torchaudio
PYTHONPATH=src python3 - <<'PY'  # fallback Mamba model construction smoke
PYTHONPATH=src python3 - <<'PY'  # fallback Mamba controller train/eval smoke
PYTHONPATH=src python3 - <<'PY'  # main.pipeline smoke with temp manifest/labels and fake torchaudio
```

結果：

- compileall 通過。
- `hubert_mamba/mamba_base_iter1`、`hubert_mamba/mamba_base_iter2` config load 通過。
- public imports 通過：`HubertMambaConfig`、`HubertMambaController`、`HubertPretrainDataset`、`HubertMambaModel`。
- source/config 搜尋已無 `deepfake`、`ASVspoof`、`Trainer`、`Tester`、`LineBot` 相關殘留。
- `GeneralConfig` 預設為 `train=True, eval=False`，並已移除無作用的 `produce_evaluation_file`。
- fake `torchaudio` pad smoke 通過，`Batch.length` 保留原有效長度。
- `mamba_ssm` 未安裝時，`TorchMambaFallback` model construction 通過。
- fallback Mamba controller smoke 通過，包含 train mode、eval mode、finite loss、checkpoint write。
- `main.pipeline()` smoke 通過，使用暫存 manifest/labels、fake `torchaudio`、CPU 小模型和 fallback mixer 完成 1 update + validation + checkpoint。
- `uv lock --check` 通過，lockfile 與 `pyproject.toml` 目前一致。

## Remaining Scope

- 未在本機安裝/編譯真實 `mamba-ssm` / `causal-conv1d` CUDA kernel；依使用者要求未嘗試編譯。
- 未以真實 `torchaudio` 音檔跑 waveform smoke。
- 未執行 LibriSpeech 960h 的 250k + 400k full schedule，因此不宣稱 paper-level result。
