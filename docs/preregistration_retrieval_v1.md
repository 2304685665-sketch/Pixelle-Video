# 预注册：镜头级素材检索 v1

> 在运行留出集评测**之前**填写并提交。提交之后，下面的内容一律不再修改；如需改动，新建 v2 文件并说明原因。

- 提交日期：
- 协议版本：`retrieval-v1`（命中规则：同一文件，且时间 IoU ≥ 0.3，或预测完全落在标注内并覆盖 ≥ 50%）
- 留出集文件：`benchmarks/labels/footage_holdout.jsonl`
- 留出集 SHA-256：
- 留出集条数：
- 素材：时长共 ___ 分钟，___ 个文件；文件名保持相机原始命名（是 / 否）
- 索引参数（在开发集上确定后固定）：threshold = ___，min_len = ___，max_len = ___，keyframes = ___，cut_method = ___
- 编码器：Chinese-CLIP ViT-B/16（`OFA-Sys/chinese-clip-vit-base-patch16`）

- VLM 模型（画面描述 / 素材描述）：
- LLM 模型（llm-asset 对照组）：

## 假设

- **H1（主假设）：** 在留出集上，镜头级混合检索（`visual+bm25-caption`）的 Hit@1 高于原版的 LLM 整文件分配（`llm-asset`）。
- **H2（粒度）：** `visual` 的 Hit@1 高于 `asset-visual`。两者视觉证据相同，唯一差别是返回粒度。

## 主要指标与检验

- 主要指标：留出集 Hit@1
- 检验方法：精确 McNemar（双侧），H1 和 H2 各做一次
- 显著性阈值：α = 0.05；两个假设用 Holm 校正
- 次要指标（只做描述，不做检验）：Hit@3、MRR、文件命中率、`llm-asset` 编造路径数、`asset-filename` 的结果

## 预期

- 我预期 `visual` 的 Hit@1 约为 ___，`asset-visual` 约为 ___。
- 如果结果不显著，我会报告：

## 不会做的事

- 不在留出集上调参，也不在留出集上做失败审计。
- 不在看到留出集结果后修改命中规则或标注。
