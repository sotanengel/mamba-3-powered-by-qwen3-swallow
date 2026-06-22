# セットアップ手順 (Docker / DevContainer)

本プロジェクトはすべての開発・学習・推論を Docker コンテナ内で完結させる。
ホスト OS には Docker Desktop と NVIDIA Container Toolkit のみ要求し、
`mamba-ssm` のソースビルドはイメージに焼き込んで再現性を確保する。

## 前提

| 項目 | 要求 |
|------|------|
| OS | Windows 11 + WSL2 (Ubuntu 22.04) |
| GPU | NVIDIA RTX 3060 Ti (Ampere, bf16 対応) |
| VRAM | 8 GB 以上 |
| Docker Desktop | 4.30+ (WSL2 backend) |
| NVIDIA Container Toolkit | WSL2 上で `nvidia-smi` が動くこと |
| ホスト RAM | 32 GB 以上推奨 |
| ディスク空き | 100 GB 以上 |

## 1. Docker Desktop と NVIDIA サポートの確認

PowerShell から:

```powershell
docker version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

`nvidia-smi` がコンテナ内で RTX 3060 Ti を表示すれば OK。

## 2. リポジトリ取得

```powershell
cd C:\Users\na-g-
git clone git@github.com:sotanengel/mamba-3-powered-by-qwen3-swallow.git
cd mamba-3-powered-by-qwen3-swallow
```

## 3. joryu パイプラインのパスを環境変数で渡す

学習データは [joryu パイプライン](https://github.com/sotanengel/qwen3-swallow-8B-RL-v0.2-AWQ-INT4-joryu-pipline) の出力を read-only でマウントして消費する。

PowerShell:

```powershell
$env:JORYU_PATH = "C:\qwen3-swallow-8B-RL-v0.2-AWQ-INT4-joryu-pipline\data\distilled"
```

WSL bash:

```bash
export JORYU_PATH=/mnt/c/qwen3-swallow-8B-RL-v0.2-AWQ-INT4-joryu-pipline/data/distilled
```

## 4. イメージのビルド (初回 30〜60 分)

```powershell
docker compose -f docker/docker-compose.yml build train
```

mamba-ssm のソースビルドがイメージ層に焼き込まれるので、以降の起動は高速。

## 5. スモークテスト

```powershell
docker compose -f docker/docker-compose.yml run --rm train python -c "from mamba_ssm import Mamba3; print('OK')"
docker compose -f docker/docker-compose.yml run --rm train python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

両方とも成功すれば環境構築完了。

## 6. VS Code Dev Container で開発

1. VS Code に `Dev Containers` 拡張を入れる
2. 本リポジトリを開き、コマンドパレットから **Dev Containers: Reopen in Container**
3. 初回のみ `post-create.sh` が `pip install -e ".[dev]"` を実行
4. `pytest tests/` が動くことを確認

## 7. テスト (CPU only / GPU)

```bash
# コンテナ内
pytest tests/                  # CPU テストのみ
pytest tests/ --run-gpu        # GPU テストも含む
ruff check .
ruff format --check .
mypy src/mamba3jp
```

## トラブルシュート

| 症状 | 対処 |
|------|------|
| `nvidia-smi` がコンテナ内で動かない | Docker Desktop の Settings → Resources → WSL Integration を有効化し、NVIDIA Container Toolkit を WSL に入れ直す |
| `mamba_ssm` の import エラー | イメージ再ビルド (`docker compose build --no-cache train`) |
| `causal_conv1d` のビルド失敗 | CUDA バージョン不一致が多い。`nvcc --version` がコンテナ内で 12.4 系であることを確認 |
| OOM (8 GB を超える) | `configs/model_50m.yaml` に切替、または `seq_len` を 1024 に短縮 |
