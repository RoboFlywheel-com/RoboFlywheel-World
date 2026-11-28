# ADR-001：RoomPlan 是 core 与房间生成之间的契约，core 冻结

- 状态：**accepted**（2026-09-21，用户决策）
- 相关：ADR-002（房间外壳切到 Infinigen）

## Context

现状：core 在**没有房间**的坐标系里解出任务单元（`solvers/cell.py` 的 `solve_cell`），
再由 `solvers/room.py:1092` `fit_room_around_cell` **画**一个房间包住 `cell.protected`，
并返回"把 core 放进房间"的平移。这条反转是有实测收益的：它把 office 杯子的可用接近方向
从 24 个里的 **2 个**提到 **19 个**，而在反转之前决定那个位置的是**房间宽度的随机种子**。

产品（`docs/PRODUCT.md` §2）要求输入含**房间规划**。调研结论是：三个参照项目都不能把户型
做成一等输入（`docs/research/comparison.md` §2.1）——SceneSmith 的尺寸由 LLM 发明、
SAGE 的入口要求交出完整内部 FloorPlan、Infinigen v1.14 根本做不到。

同时，ADR-002 决定把外壳切到 Infinigen，而 Infinigen 需要一个**米制户型**作为输入
（`Solver.floor_plan`）。于是需要一样东西，把 core 的结论翻译成 Infinigen 能吃的户型。

## Evidence

- `fit_room_around_cell` 已经在做这件事的一半：它读 `cell.protected`、按 `wall_standoff_m`
  算出所需宽深、从 spec 的 `size_bounds_m` 采样形状（family / notches），并处理门与接近弧
  （`door_bearing_deg`、`door_in_approach_arc`）。产物是 `RoomShell` + 平移。
- `RoomShell`（`contracts/state.py:100`）已带 `source: str`，是一个现成的来源标注位。
- 参照项目的教训：SceneSmith 的 plan 是**采样**（尺寸由 LLM 在 1.5–20 m 包络内发明），
  它整条"可恢复"设计正是为了不重新采样；SAGE 的 layout JSON 是唯一枢纽文档且可往返；
  Infinigen 新树接受米制 shapely 户型、并**跳过整个退火搜索**（确定性执行）。
- 三家都没有"core → plan → 房间生成"这个接缝。

## Options

1. **plan 只作输出**（现状）：房间继续由 `fit_room_around_cell` 拟合，外部无法给户型。
2. **plan 是一等外部输入**：外部给户型，core 塞进去；core-first 变成次要路径。
3. **core 冻结，plan 成为 core → 房间生成之间的契约**（选中）：core 解出后，
   **拟合阶段输出一份 floor plan**，房间生成消费它建房；core 此后不再因房间而变。

## Decision

选 **3**。

具体形状：

```
solve_cell(spec)                    # 不变，无房间坐标系
    ↓  core（冻结组件）
emit_floor_plan(core, spec)         # 由现在的 fit_room_around_cell 演进而来
    ↓  RoomPlan IR（米制，含开口）
room generation                     # 消费 plan 建房（当前实现见 ADR-002）
    ↓  外壳几何
furnish / layout / gates            # 不变
```

`RoomPlan` 的内容（最小集）：房间多边形（米，声明单位与 up-axis）、开口（门/窗：贴墙线段 +
宽/高/sill/类型）、地面与墙面材质引用、core 的平移量与 `protected` 区域、以及一个
`source` 字段标明 plan 由谁产出。

## Rationale

- **core 冻结是这条路径的价值所在**：任务的位置由任务决定，不由房间反推。
  一旦房间生成改成"消费 plan"，就必须切断"房间可以反过来改 core"这条回路，
  否则反转的实测收益会被重新引入的耦合吃掉。
- **plan 作为契约让"外部给户型"成为同一接口的另一条来源**：产品要求的输入路径因此
  不需要另起一套，也不与 core-first 冲突（缺省仍由 core 产出 plan）。
- 这正是三家参照都没有的接缝，也是我们能同时满足"任务优先"与"用户可给户型"的唯一位置。

## Consequences

**变了的**：

- `fit_room_around_cell` 的角色改变：从"画出房间并返回 `RoomShell`"变成"产出 plan"。
  `RoomShell` 仍然存在（下游 gate 与 Isaac worker 都消费它），但它的 `source` 必须能区分
  plan 的来源（core 产出 / 外部给 / 未来其它）。
- 需要定义 RoomPlan IR 并**冻结哈希**：下游把它当冻结、已哈希的输入消费
  （借 SceneSmith 的 `resume_from_path` 性质），这样 A/B 比较资产库时 plan 是**夹具**而非采样。

**新成本（本 ADR 明确接受）**：

- **必须有一条 gate：建出来的外壳 == 我们给的 plan。** Infinigen 不回吐几何（房间多边形序列化成
  `"<not-serialized>"`，见 ADR-002），所以"我们给了什么"与"它建了什么"之间没有证据，
  除非去**测量建成的网格**（或先 patch 导出）。这条 gate 是采用 Infinigen 的连带成本，
  在这里记为不变量而不是实现细节。

**被推迟的**：

- 外部输入户型（选项 2 的能力）排期后置；本 ADR 只保证接口的形状让它可接。
- plan 的编辑能力（部分覆盖、"外壳用我的、家具你重排"）不在本期。

**未决**：

- RoomPlan 是否复用 `schemas/` 的 schema 发射机制（`rfw_w.scenes.preparation.emit_schemas` 的 REGISTRY）。
- plan 的版本字段与向后兼容策略。