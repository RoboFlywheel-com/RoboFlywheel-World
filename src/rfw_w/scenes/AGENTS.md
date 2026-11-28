# RFW-W 场景模块

遵循根 [AGENTS.md](../../../AGENTS.md)。接口变更先读
[场景架构](../../../docs/scenes/ARCHITECTURE.md) 与相关决策。

- 保持任务核心 → 房间 → 外围的顺序，区分初态与目标态；变更使相关证据失效。
- application 编排生成与修复，assembly 负责布局，delivery 组合证据。
- 配置、schema 与示例从安装包资源加载；worker 运行安装后的模块。
- 修改资产接入时运行 `bash scripts/dev/test-scenes.sh --assets --require-assets`。
  `--delivery` 检查已有交付；`--stdlib-only` 检查核心导入。
