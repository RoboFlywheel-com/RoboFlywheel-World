# ADR-015：任务核心、背景与干扰物共用一份全屋图

- 状态：**accepted，SPEC-022 契约验收通过**（2026-10-03）。
- 范围：跨房 G0–G3 新路径；不改旧单房 `SceneIR` 的既有语义。

## Context

SPEC-019 的逐房树只允许 `priority=optional` 的资产角色作为 `distractor`；G3 首例因此只有两件台面干扰物。用户现在要求 Agent 修改第一版核心场景图，并增加地面家具、墙面安装物，同时保留任务关系的可追溯性。如果另存一份可编辑的背景图，房间、门、任务工作区和资产实例会出现两套互不约束的事实。

## Evidence

- `contracts/task_world_graph.py::_validate_composition` 目前把所有可选角色限定为 `task_role=distractor`，且 `physical_class` 仅接受 `furniture|manipuland`。
- `agents/task_world_assembly.py` 为两房各造一个可选台面角色；`assets/room_tools.py` 只按图中角色检索与摆放。
- SPEC-020 的建成墙保存在 OBJ 与几何账中；图中的 `wall_set` 仍是待求解意图，尚无可安装的墙面锚点。
- ADR-012/014 已确定一份全屋图与只读逐房视图的边界。

## Options

1. 单独维护背景/装饰图，在交付时合并。
2. 把所有非核心物体都继续叫 `distractor`。
3. 保留一份全屋图，为角色和实例增加与任务必要性正交的用途标签，并引用建成几何锚点（拟采用）。

## Decision

采用 3。任务核心 `task_core`、通常不参与任务的 `background`、有意构成干扰的 `distraction` 都是**同一张 `TaskWorldGraph` 的节点**，共用稳定 ID、房间归属、生成树与图版本。`priority=core|optional` 仍表示是否为必需角色；用途标签解释它为何在场景中，不代替几何支撑关系或任务状态谓词。背景/干扰角色不得被悄悄写入任务目标、动作前后态或必需验证义务。真实实例继承角色用途，并在同一图中通过 `role_bound_to` 绑定。

对于后续布置，地面、台面和墙面的承载/安装锚点引用 G2/G3 的建成记录；墙面物件还要记录墙段、面向房间的一侧和安装姿态。语义图不自行证明墙段存在或机器人可通过。

## Rationale

一份权威图允许同一验证器同时检查核心任务、可选布置和它们之间的冲突。`background` 与 `distraction` 的区别让用户、Agent 和验证器都能辨认“环境真实感”与“任务干扰”；它们都不能自动改变用户给出的任务。

## Consequences

- SPEC-022 定义新 composition 版本、迁移与负例；旧 v1 图仍可读取，不被静默升级。
- SPEC-023 的 Agent 图修改以任务锚点和用户硬约束为界；实际几何与资产仍由工具求解。
- SPEC-024/025 分别验证地面与墙面布置；新增物体进入同一资产账和图版本。
- 未实现与未验收之前，本 ADR 不改变 `ARCHITECTURE.md` 对当前系统的描述。
