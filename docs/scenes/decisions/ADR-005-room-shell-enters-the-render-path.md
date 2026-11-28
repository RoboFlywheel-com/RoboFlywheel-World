# ADR-005：建成的外壳进入渲染路径（修订 SPEC-001 R1）

- 状态：**accepted**（2026-09-21，用户决策）
- 相关：[ADR-002](ADR-002-room-shell-from-infinigen.md)（外壳改由 Infinigen 生成）、
  SPEC-005（本 ADR 的实现）、
  SPEC-004（gate 与渲染必须是同一个几何）
- **读正文前须知**：本文写于 SPEC 拆分之前，那时"SPEC-001"是一个覆盖整条房间线的 SPEC。
  正文里的 **`SPEC-001 R1` 现在是 SPEC-005 R1**，
  `SPEC-001 §2 的输出表` 现在是 SPEC-005 §2，`SPEC-001 §8.6` 现在是 SPEC-005 §8.5。
  正文按原样保留，不改写历史。

## Context

ADR-002 决定把房间外壳从自建切到 Infinigen，动机写在它的 Context 里：**现行外壳是我们
在 Isaac 里直接构造的**（`workers/isaac/render_worker.py:171` `_author_room`），用户判定为
"简陋"——**判的是渲染出来的那张图**。

SPEC-001 R1 把这个动机接丢了。它写的是"`RoomGeometry` 与所有 gate 的输入不变"，而 §2 的输出表
把 Infinigen 几何的消费者写成**交付物**。两者合起来意味着：切完 build 路径之后，交付物里躺着
Infinigen 的 `.usdc` 与烘焙贴图，**而用户看到的那张图仍然是今天这套程序化外壳**。

这不是可以绕过的实现细节，因为下游根本没有第二条路：

- `RoomGeometry`（`workers/isaac/protocol.py:142`）只有 `polygon_xy` / `height_m` /
  `floor_z_m` / `openings` 四个字段，**没有 URI、没有来源、没有"加载已建成外壳"的入口**。
- worker 从不打开现成 USD：`Usd.Stage.Open` / `AddReference` / `Sdf.Reference` 在
  `workers/isaac/` 下**一个都 grep 不到**；房间的每一个 prim 都由
  `triangulated_floor` + `build_wall` 从多边形现搓。
- `RoomShell.source`（`contracts/state.py:121`）本来是现成的来源标注位，但在
  `run_m1_acceptance.py:2354` 翻译成 `RoomGeometry` 时**被丢掉了**，worker 无从分辨外壳来自谁。

## Evidence

- 上述三处代码事实（本轮检查）。
- **探针实测（2026-09-21，SPEC-001 Step 0）**：office 的 L 形单房间 plan 能被 Infinigen 建成，
  `-t coarse` 2 min 52 s，房间多边形 / 层高 / 墙厚**逐位复现**（建成包围盒
  4.4524 × 3.5598 × 2.9697，与 plan 相同）。也就是说：**值得渲染的那部分质量确实存在**，
  只是按 R1 的原文进不了渲染。
- 同时实测：开口尺寸不由 plan 决定（见 SPEC-001 §5 与 R10）。

## Options

1. **维持 R1 原文**：Infinigen 几何只进交付物，Isaac 继续从多边形现搓。渲染观感不变，
   用户看到的仍是今天这套；`shell.matches_plan` 去量一个**渲染器从来不显示**的网格。
2. **修订 R1**（选中）：`RoomGeometry` 加一个可选 `shell_uri`，`_author_room` 多一条引用分支。
3. **把 Infinigen 几何反解回 polygon + openings**，worker 照旧现搓。等于丢掉 Infinigen 的
   建筑细部与程序化材质——两头不靠。

## Decision

选 **2**。

```python
RoomGeometry:
    polygon_xy, height_m, floor_z_m, openings
  + shell_uri: str | None = None      # 有值时不再现搓墙地顶

render_worker._author_room:
    shell_uri 有值 → 引用建成几何
    否则          → triangulated_floor + build_wall（今天的行为，逐字保留）
```

**分支是可选且向后兼容的**：`shell_uri` 缺省 `None`，`builtin` provider 与所有现有交付走原路径，
行为不变。这不是"换掉渲染器"，是让"外壳从哪来"成为 `RoomGeometry` 能表达的一件事。

## Rationale

- **ADR-002 的目的就是那张图。** 不修订 R1，ADR-002 的采纳理由不成立：我们付了 Infinigen 的成本
  （分钟级生成、烘焙管线、上游冻结绑定），却拿不到它换来的东西。
- **gate 与渲染必须是同一个几何。** ADR-001 要求 `shell.matches_plan` 这条 gate。如果渲染用 A、
  gate 量 B，那这条 gate 通过与否都不说明交付物是对的——它测的不是用户看的东西。
- **修订的成本是局部的**：一个可选字段 + `_author_room` 的一条分支。相较之下，选项 1 让整条
  ADR-002 白做，选项 3 让 Infinigen 只剩"生成一个和我们自己建得差不多的多边形"。

## Consequences

**变了的**：

- `workers/isaac/protocol.py::RoomGeometry` 增加 `shell_uri`（`to_dict`/`from_dict` 一并），
  `WorkerReport` 需要记录这一趟是引用还是现搓，供独立验证区分。
- `_author_room` 增加引用分支；引用路径下 `openings_cut` 不再是"我们切了几个洞"，
  而是"建成几何里量到几个洞"——`assert_self_consistent` 的语义要跟着说清楚。
- SPEC-001 §8.6 的"下游零改动"不再成立于 `RoomGeometry` 与 `_author_room`。

**新成本（本 ADR 明确接受）**：

- **`shell_uri` 把一条外部路径接进了 worker**。引用一个文件意味着 worker 的正确性依赖那个文件
  存在且内容对——交付必须把几何一起打包，独立验证（`rfw_w.scenes.validation.verify_delivery`）要能
  在不跑 Isaac 的情况下确认该 URI 可解析。
- **Isaac 里是烘焙贴图，不等于 Blender 渲染。** ADR-002 已经记过这条；引用路径不会改善它，
  只会让它第一次显形。

**未决**：

- `shell_uri` 指向 `.usdc` 还是别的格式、以及材质是内嵌还是并列 `material_uris`——
  属 ADR-002 的烘焙管线实现，不在本 ADR。
- 引用路径下 `RoomMaterialsRequest` 是否还有意义（材质由建成几何自带），或退化为回退路径的参数。