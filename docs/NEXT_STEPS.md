# 接下来怎么做（按顺序）

全部命令都在你 **Mac 自己的终端** 里运行（不是 Claude 的沙盒），仓库在 `~/Desktop/Pixelle-Video`。

---

## 第 0 步 · 环境（约 20 分钟，一次性）

```bash
brew install ffmpeg redis uv
cd ~/Desktop/Pixelle-Video
git pull
uv sync --extra dev --extra editing --extra editing-models --extra serving
uv run pytest -q          # 应该全部通过（Mac 上若没启动 redis，serving 测试会自动跳过）
```

---

## 第 1 步 · 测真实错误分布（约 1–2 小时，会产生 API 费用）

**目的：** 把可靠性实验里的故障比例从“占位值”换成实测值。之后简历才能写“基于真实错误分布的故障注入”。

1. **配置 API 密钥。** 运行 `uv run streamlit run web/app.py`，在「⚙️ 系统配置 → API 媒体模型配置」里填密钥；也可以把 `config.example.yaml` 复制成 `config.yaml` 后直接编辑。
2. **选一个最便宜的视频模型。** 先列出可选的模型，再到控制台确认单价：
   ```bash
   uv run python benchmarks/measure_provider_errors.py --list
   ```
3. **准备 prompt 文件** `benchmarks/prompts/real_prompts.txt`。一行一条，写 30 条左右。直接用你真实脚本（比如《家》）里每个镜头的画面描述，要包括情绪化、抽象的句子，这样测到的审核拒绝率才真实。
4. **先小规模跑 10 次**，确认能跑通，也心里有数要花多少钱：
   ```bash
   uv run python benchmarks/measure_provider_errors.py --workflow api/dashscope/wan2.6-i2v-flash \
     --prompts benchmarks/prompts/real_prompts.txt --n 10 --concurrency 2 \
     --media-type video --image <任意一张竖版图片.png> --out benchmarks/results/provider_errors_trial
   ```
   文生视频（t2v）模型不需要 `--image`。
5. **正式跑 60 次**，并发设为 4，因为 429 限流和并发数有关：
   ```bash
   ... --n 60 --concurrency 4 --out benchmarks/results/provider_errors_v1
   ```
6. **用实测分布重跑消融实验**（约 4 分钟），然后把 `docs/RESUME_MAPPING.md` 第 1 节表格里的数字换成新结果：
   ```bash
   uv run python benchmarks/reliability_ablation.py --videos 500 --fail 0.1 0.3 \
     --mix-file benchmarks/results/provider_errors_v1.json \
     --out benchmarks/results/reliability_ablation_measured.json
   ```

如果 60 次里一次失败都没有，说明这个供应商在这个并发下很稳定。这本身就是一个结论：简历上写“实测失败率 X%，故障注入压测 10% / 30%”。

---

## 第 2 步 · 检索实验（标注约 1–2 天，是最花时间也最有价值的一步）

**目的：** 在冻结留出集上证明“镜头级检索”比“整文件分配”好，并给出 p 值。这一步对标参照项目里“52 条冻结留出查询上 Hit@1 0.808 → 0.962”的写法。

1. **下载模型**（约 750 MB）：
   ```bash
   HF_ENDPOINT=https://hf-mirror.com uvx --from huggingface_hub hf download \
     OFA-Sys/chinese-clip-vit-base-patch16 --local-dir models/chinese-clip-vit-base-patch16
   ```
2. **整理素材。** 准备 10–20 分钟、至少 15 个文件，放进一个**仓库外**的文件夹，保留相机原始文件名。
3. **建索引，打开预览页：**
   ```bash
   uv run python benchmarks/retrieval_eval.py index --footage ~/Movies/eval_footage \
     --out runs/footage_idx --model-dir models/chinese-clip-vit-base-patch16
   uv run python benchmarks/retrieval_eval.py sheet --index runs/footage_idx
   open runs/footage_idx/contact_sheet.html
   ```
4. **写 100 条标注，先写完再看任何检索结果。** 格式见 `docs/editing/README.md`。
   - 拆成两个文件：`benchmarks/labels/footage_dev.jsonl`（40 条，用来调参）和 `benchmarks/labels/footage_holdout.jsonl`（60 条，冻结留出集）。
   - 留出集写完后立刻 `git commit`，再运行 `shasum -a 256 benchmarks/labels/footage_holdout.jsonl`，把哈希值填进预注册文档。之后不能再改。
5. **填写预注册** `docs/preregistration_retrieval_v1.md`，然后提交。
6. **在开发集上调参并做失败审计**（见第 3 步），可以调切分阈值、`--max-len`、关键帧数量、融合权重。
7. **最后在留出集上只跑一次：**
   ```bash
   uv run python benchmarks/retrieval_eval.py eval --index runs/footage_idx \
     --labels benchmarks/labels/footage_holdout.jsonl --model-dir models/chinese-clip-vit-base-patch16 \
     --methods asset-filename asset-visual visual --baseline asset-visual \
     --out benchmarks/results/retrieval_holdout_v1.json
   ```
   主对比是 `visual` 对 `asset-visual`：两者用完全相同的视觉证据，唯一差别是粒度。

---

## 第 3 步 · 失败审计（对标“三轮失败审计修复评分器误判”）

每一轮都这样做：

1. 在**开发集**上运行 eval，打开结果 JSON，把 `per_query` 里 `first_hit_rank` 为空或大于 1 的查询全部看一遍。
2. 每个失败归到一类，写进 `docs/audit_log.md`：
   - **标注错**：标的时间段不对、漏了也合适的镜头 → 修标注，或者加 `alts`。
   - **规则误判**：检索结果其实合适，但命中规则判成了没中 → 改规则，并把协议版本号升到 `retrieval-v2`。
   - **切分错**：切点把一个动作切断了，或者两个镜头没切开 → 调切分参数。
   - **检索真错**：保留，作为后续改进的依据。
3. 记录每一轮修了多少条、指标怎么变化。做 2–3 轮。**留出集不参与审计**，否则就不再是留出集。

---

## 第 4 步 · 之后可以继续加的（需要再写代码）

- **画面描述通道：** 用 VLM 给每个片段生成一句画面描述，写进索引的 `caption` 字段，再用 `visual+bm25-caption` 融合检索。用来解决“家似乎不再是我想象的模样”这类抽象旁白。原作者的 `services/api_services/vlm_client.py` 可以直接调用。
- **内容安全预检：** 生成前用规则加 LLM 两层判别。构造约 100 条会触发审核的 prompt 和约 200 条正常 prompt，报告真实 API 拒绝率的下降和合法 prompt 的误拦率。
- **真实的 LLM 分配对照组：** 复现原作者 `AssetBasedPipeline` 的 LLM 直接分配，作为检索实验的第 4 个对照组。

这三项可以之后再开 Claude Code 让它照着这份文档继续做。
