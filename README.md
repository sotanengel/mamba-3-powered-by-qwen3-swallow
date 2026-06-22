# Mamba-3-JP

Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4 を教師とした日本語小型 Mamba-3 モデル
(130M / 50M) の事前学習パイプライン。RTX 3060 Ti (8 GB) で動作する。

実装はすべて Docker コンテナ内で完結し、各モジュールは TDD (テスト先行) で書かれている。

- 計画 (EPIC): [#1](https://github.com/sotanengel/mamba-3-powered-by-qwen3-swallow/issues/1)
- 教師モデル: [tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4](https://huggingface.co/tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4)
- 学習データ: [joryu パイプライン](https://github.com/sotanengel/qwen3-swallow-8B-RL-v0.2-AWQ-INT4-joryu-pipline) の `responses.jsonl` を **そのまま消費**する

## アーキテクチャ

| 構成 | d_model | n_layer | mixer | 用途 |
|------|---------|---------|-------|------|
| `configs/model_130m.yaml` | 768 | 24 | Mamba3 (SISO) | 主構成 |
| `configs/model_50m.yaml` | 512 | 16 | Mamba3 (SISO) | フォールバック / スモーク |
| `configs/model_mamba2_130m.yaml` | 768 | 24 | Mamba2 | ベースライン |

トークナイザは Qwen3-8B (語彙 ~151k) を再利用し、`tie_embeddings=True` で
埋め込み層と出力層を共有する。

## セットアップ

詳細は [docs/setup.md](docs/setup.md)。要点のみ:

```powershell
# Windows PowerShell
$env:JORYU_PATH = "C:\qwen3-swallow-8B-RL-v0.2-AWQ-INT4-joryu-pipline\data\distilled"
docker compose -f docker/docker-compose.yml build train
docker compose -f docker/docker-compose.yml run --rm train python -c "from mamba_ssm import Mamba3; print('OK')"
```

VS Code Dev Containers から **Reopen in Container** で開発できる。

## パイプライン

### 0. テスト (TDD)

```bash
pytest tests/                  # CPU テストのみ (デフォルト)
pytest tests/ --run-gpu        # GPU を含む全テスト
ruff check . && mypy src/mamba3jp
```

### 1. 取り込み: joryu → ChatML

```bash
python scripts/ingest_joryu.py \
    --input  /data/joryu/responses.jsonl \
    --output data/intermediate/chatml.jsonl \
    --stats  logs/ingest_stats.json
```

`thinking_trace` がある場合は assistant ターン冒頭に `<think>...</think>`
として再挿入される (`--no-thinking` で抑制可)。品質フィルタは
[要件 4.5](docs/setup.md) に準拠。

### 2. トークナイズ: ChatML → binidx

```bash
python scripts/tokenize_data.py \
    --input   data/intermediate/chatml.jsonl \
    --out-dir data/tokenized \
    --tokenizer Qwen/Qwen3-8B \
    --val-ratio 0.05
```

### 3. モデル構築 (Mamba3 ディスパッチ確認)

```bash
python scripts/build_model.py --config configs/model_130m.yaml --smoke-forward
python scripts/build_model.py --config configs/model_50m.yaml  --smoke-forward
python scripts/build_model.py --config configs/model_mamba2_130m.yaml --smoke-forward
```

### 4. 訓練

```bash
python scripts/train.py \
    --model configs/model_130m.yaml \
    --train configs/train.yaml \
    --data  configs/data.yaml \
    --out   /checkpoints/mamba3-130m

# Resume:
python scripts/train.py ... --resume /checkpoints/mamba3-130m/last.pt
```

Mamba-2 ベースライン:

```bash
python scripts/train.py --model configs/model_mamba2_130m.yaml ...
```

### 5. 評価

```bash
python scripts/evaluate.py \
    --ckpt  /checkpoints/mamba3-130m/best.pt \
    --model configs/model_130m.yaml \
    --data  configs/data.yaml \
    --tasks lambada_openai,jcommonsenseqa
```

### 6. 推論

```bash
python scripts/generate.py \
    --ckpt   /checkpoints/mamba3-130m/best.pt \
    --model  configs/model_130m.yaml \
    --prompt "日本の四季について簡潔に教えてください。" \
    --max-new-tokens 512 --temperature 0.7 --top-p 0.9
```

## TDD 方針

| モジュール | テスト | 状態 |
|------------|--------|------|
| `src/mamba3jp/data/ingest.py` | `tests/test_ingest.py` | 11 / 11 ✓ |
| `src/mamba3jp/data/binidx.py` | `tests/test_binidx.py` | 6 / 6 ✓ |
| `src/mamba3jp/data/dataset.py` | `tests/test_dataset.py` | 5 / 5 ✓ |
| `src/mamba3jp/model/builder.py` | `tests/test_builder.py` | 10 / 10 (+5 GPU) ✓ |
| `src/mamba3jp/train/checkpoint.py` | `tests/test_checkpoint.py` | 7 / 7 ✓ |
| `src/mamba3jp/train/loop.py` | `tests/test_loop_smoke.py` | 4 / 4 ✓ |

GPU テストは `@pytest.mark.gpu` でマークされ、`--run-gpu` でのみ実行される。

## フォールバック

| 状況 | 対処 |
|------|------|
| `mamba-ssm` ソースビルド失敗 | `configs/model_mamba2_130m.yaml` で Mamba-2 にフォールバック (パイプライン検証は継続) |
| 130M 訓練が不安定 (NaN / 損失発散) | `configs/model_50m.yaml` に切替、lr を 1/10 に |
| 合成データ不足 | `--max-steps` を増やしエポック数を稼ぐ / joryu 側に追加生成依頼 |
| 370M で VRAM 8GB 超過 | seq_len を 1024 に短縮、または 130M に留める |

## 定性サンプル

学習完了後にここに 5 件の生成結果を貼る。

## ライセンス

Apache 2.0 ([LICENSE](LICENSE))

## 謝辞

- [state-spaces/mamba](https://github.com/state-spaces/mamba) — Mamba-3 アーキテクチャ
- [tokyotech-llm/Qwen3-Swallow](https://huggingface.co/tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4) — 教師モデル
- [sotanengel/qwen3-swallow-8B-RL-v0.2-AWQ-INT4-joryu-pipline](https://github.com/sotanengel/qwen3-swallow-8B-RL-v0.2-AWQ-INT4-joryu-pipline) — データ生成パイプライン
