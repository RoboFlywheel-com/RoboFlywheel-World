# ADR-010：人工选材的论文 case 使用独立图用装配与证据

- 状态：accepted for SPEC-012 scope（2026-09-23，依据本次用户明确的人工选材与三类型展示要求）。

## Context

用户要求先拆 SPEC，用真实 case 收集论文管线素材，软体、铰链与刚体共同进入 core，机器人入景；
不需要真实检索。生产 M1 仍为刚体在线闭环，ADR-004 不覆盖软体检索。

## Evidence

- `rfw_w.scenes.commands.generate:1596` 从刚体库选择一个任务目标；
  `src/rfw_w/scenes/agents/scene_compiler.py:208` 拒绝非刚体输入。
- `src/rfw_w/scenes/workers/isaac/protocol.py:30` 的 MeshInstance 并非通用机器人/关节 USD 装配协议。
- v2 的 `scripts/figure/render_isolated_asset.py:203` 用 wrapper 引用非刚体包；
  `src/pvscene/robot_placement.py:16` 起定义现成机器人源。
- 初查实际存在的候选及限制见 SPEC-012 PLAN 实施记录。

## Options

1. 扩展生产多目标、多物理与机器人执行路径，再制作图。
2. 在图用路径固定真实资产与布局，复用房间/建壳能力，独立记录素材与证据。
3. 拼接旧图或生成示意三维图。

## Decision

采用 2。图用 case 可人工指定源资产、静态姿态与支撑关系，先冻结再拟合房间。
用 `case_core_hash` 标识展示冻结范围，不冒充生产 `core_hash`；机器人也在冻结与占位范围内。
所有图片来自同一新 case；真实 Isaac 渲染、旧资产验证和未运行动态检查分别记账。

## Rationale

直接满足用户的素材任务，并使实现规模限定在可独立验收的素材包。
选材展示无需等待生产多物理执行，但不能因此声称执行能力已经完成。

## Consequences

需新增图用配置、装配/渲染/审计入口；它们尚未实现。
生产 pipeline、能力表和 readiness 不变；最终主图须标注 curated example 与未运行状态。
软体/铰链动态接线及机器人运动学若以后需要，另立 SPEC。
