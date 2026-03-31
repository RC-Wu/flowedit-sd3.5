# 建仓说明（v1）

目标：在空仓内建立可持续迭代的 `flowedit-sd3.5` 基础结构，优先承载 `2D SD3.5 FlowEdit + DNAEdit`，并预留 `3d_smoke` 诊断入口。

## 本次完成内容

- 创建目录骨架：
  - `src/flowedit_sd35/`
  - `benchmarks/2d/`
  - `benchmarks/3d_smoke/`
  - `docs/`
  - `scripts/`
- 增加最小 package 元数据：`pyproject.toml`
- 增加最小 runner stub：
  - `src/flowedit_sd35/runner.py`
  - `scripts/run_2d_stub.py`
- 增加 case/config 模板：
  - `benchmarks/2d/cases/dnaedit_sd35_minimal.yaml`
  - `benchmarks/2d/configs/flowedit_sd35_base.yaml`
  - `benchmarks/3d_smoke/cases/sd35_3d_smoke_minimal.yaml`
  - `benchmarks/3d_smoke/configs/sd35_3d_smoke_base.yaml`
- 增加来源映射文档：`docs/ASSET_PROVENANCE.md`
- 增加上游资产同步脚本：`scripts/sync_assets_from_editsplat.py`

## 为什么先做 stub

- 当前会话中 `/dev_vepfs/...` 路径不可见，无法安全直接复制上游文件。
- 先落稳定仓结构与配置模板，可保证后续迁移时不破坏目录组织。
- 同步脚本已给出目标映射，路径可见后可直接执行批量导入。

## 后续建议顺序

1. 在可访问 `/dev_vepfs/...` 的环境执行 `scripts/sync_assets_from_editsplat.py`。
2. 先打通 2D 最小链路：将 `runner.py` 从 stub 接到 SD3.5 FlowEdit + DNAEdit 真正入口。
3. 为 `benchmarks/2d` 增加至少一个端到端 smoke test（输入图 + 配置 + 输出校验）。
4. 将 `3d_smoke` 保持为诊断辅助，不阻塞 2D 主线迭代。
