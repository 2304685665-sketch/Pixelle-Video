# 镜头级素材检索（智能剪辑 · 内核 1）

## 要解决的问题

上游 `AssetBasedPipeline` 让 LLM 直接把整个素材文件分配给脚本场景
（`pipelines/asset_based.py`：“LLM directly assigns assets to scenes - no complex matching logic needed”）：

- 粒度是整个文件，一段 3 分钟的实拍不会被切出最合适的几秒；
- 素材一多，LLM 只能看文件名和摘要猜；
- 路径写错时先按文件名模糊匹配，失败就分配一个备用素材，静默配错；
- 没有任何指标衡量“配得对不对”。

`pixelle_video/editing/` 用一个可度量的检索步骤替代它：

1. **镜头切分**（`segments.py`）：ffmpeg 低帧率采样缩略图，按相邻帧 RGB 颜色直方图的总变差检测硬切；
   过短的镜头并入前一段，过长的连续镜头按 `max_len` 等分。
   不默认使用 ffmpeg 自带的 `scene` 分数，因为它主要看亮度，会漏掉亮度相近、颜色不同的切换（纯红→纯绿打分约为 0）。
2. **编码**（`encoders.py`）：Chinese-CLIP，从本地目录加载权重。
3. **索引**（`index.py`）：片段、关键帧、embedding、文本字段（文件名，以及后续加入的画面描述、语音转写）。
4. **检索**（`retrieval.py`）：视觉通道（旁白文本 vs 关键帧，取最大值）+ BM25 文本通道（jieba 分词），RRF 融合。
   `AssetLevelBaseline` 复现上游的“整文件”粒度，作为对照组。
5. **评测**（`evaluate.py`）：时间片段命中规则、Hit@k、MRR、文件命中率、精确 McNemar 配对检验。

## 一次性准备（在你自己的 Mac 终端里执行）

```bash
cd ~/Desktop/Pixelle-Video
brew install ffmpeg            # 如果还没有
uv sync --extra dev --extra editing --extra editing-models

# 下载 Chinese-CLIP 权重（约 750 MB；国内用镜像）
HF_ENDPOINT=https://hf-mirror.com uvx --from huggingface_hub hf download \
  OFA-Sys/chinese-clip-vit-base-patch16 --local-dir models/chinese-clip-vit-base-patch16
```

`models/` 和 `runs/` 已加入 `.gitignore`，权重和索引不会被提交。

## 实验流程

### 1. 整理素材

把评测素材放进一个文件夹（不要放进仓库），可以有子文件夹。视频支持 mp4/mov/m4v/mkv/avi/webm，图片支持 jpg/png/webp。
**文件名保持相机原始命名**（如 `IMG_1234.MOV`），这样才是公平的对照：真实素材库里文件名通常没有信息量。

### 2. 建索引并生成预览页

```bash
uv run python benchmarks/retrieval_eval.py index \
  --footage ~/Movies/eval_footage --out runs/footage_idx \
  --model-dir models/chinese-clip-vit-base-patch16

uv run python benchmarks/retrieval_eval.py sheet --index runs/footage_idx
open runs/footage_idx/contact_sheet.html
```

### 3. 写标注（先写标注，再看任何检索结果）

`benchmarks/labels/footage_v1.jsonl`，每行一句旁白：

```json
{"id": "q01", "query": "我还记得守在电话旁边等他们打来", "asset": "home/IMG_1234.MOV", "start": 12.0, "end": 16.5}
```

- `asset` 是相对素材根目录的路径；`start`/`end` 是你作为剪辑者认为最合适的那段（秒）。
- 有多个都合适的镜头时加 `"alts": [{"asset": ..., "start": ..., "end": ...}]`。
- 图片素材只写 `asset`。
- 旁白要像真的口播稿，不要写成画面描述（“一部红色电话”），否则会高估视觉检索。
- 建议 40–60 句；预留 30% 作为冻结留出集，调参只看另外 70%。

### 4. 评测

```bash
uv run python benchmarks/retrieval_eval.py eval \
  --index runs/footage_idx --labels benchmarks/labels/footage_v1.jsonl \
  --model-dir models/chinese-clip-vit-base-patch16 \
  --methods asset-filename asset-visual visual \
  --baseline asset-filename --out benchmarks/results/retrieval_v1.json
```

## 命中规则（protocol `retrieval-v1`，实验前固定）

预测与标注在同一文件中，且满足其一：

- 时间 IoU ≥ 0.3；或
- 预测完全落在标注区间内，且长度 ≥ 标注长度的 50%。

图片只要求文件一致。改规则需要升协议版本号，旧结果不可混用。

## 对照组的含义

| 方法 | 含义 |
|---|---|
| `asset-filename` | 只凭文件名选整文件（近似 LLM 看文件列表时的信息） |
| `asset-visual` | 用和 `visual` 完全相同的视觉证据选文件，但返回整文件 —— 单独度量“整文件粒度”的损失 |
| `visual` | 镜头级视觉检索 |
| `visual+bm25-caption` 等 | 加入画面描述 / 语音转写通道后 RRF 融合（后续步骤） |

## 已知限制

- 上游真正的 LLM 分配作为对照组还没接入（需要 LLM API），目前用 `asset-*` 近似；正式报告前要补上。
- 硬切检测对淡入淡出等渐变转场不敏感。
- Chinese-CLIP 对抽象、情绪化的旁白（“家似乎不再是我想象的模样”）理解有限，这正是后续加入画面描述通道要解决的问题。
