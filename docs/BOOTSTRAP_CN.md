# Bootstrap 说明（纯 2D）

目标：将仓库基线固定为 `bootstrap -> minimal real 2D path`，仅保留 SD3.5 FlowEdit + DNAEdit 的 2D 叙事与示例。

## 本次收口内容

- 保留并强化纯 2D 基准目录：`benchmarks/2d/`
- 新增真实 2D 最小示例：
  - `benchmarks/2d/cases/dnaedit_sd35_real_minimal.yaml`
  - `benchmarks/2d/configs/flowedit_sd35_real_minimal.yaml`
- 新增轻量本地测试：`tests/test_minimal_real_2d_contract.py`
  - 校验 case/config manifest 契约
  - 校验 runner 入口契约
  - 校验文档为纯 2D 叙事
  - 校验 benchmark 不包含非 2D 目录

## 验证要求（必须长期保持）

- 每个版本至少保留一个可本地执行的 2D smoke 契约检查路径（不要求拉起模型）。
- 每个版本 smoke 必须记录显存用量（VRAM usage）。
- 显存指标至少包含：
  - `peak_vram_mb`
  - `allocated_vram_mb`
  - `reserved_vram_mb`
- 上述要求必须在 benchmark 配置的 `validation.smoke` 字段显式声明。

## 本地验证命令

```bash
python -m unittest -v tests.test_minimal_real_2d_contract
```
