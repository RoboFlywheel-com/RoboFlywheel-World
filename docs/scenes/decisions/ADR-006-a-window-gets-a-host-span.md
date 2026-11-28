# ADR-006：窗口拿到的是一段墙，不是一个宽度（修订 SPEC-001 R7 与 SPEC-004 的比较集）

- 状态：**accepted**（2026-09-22，用户决策）
- 相关：SPEC-001（plan 契约与它的序列化 R7）、
  SPEC-002（`_segment_for` 与抽屉）、
  SPEC-004（gate 的比较集）
- 触发：SPEC-004 §8 的第一次真跑
  ——gate 抓到了三个实验里有两个的窗口**根本没被建**。

## Context

`shell.matches_plan` 第一次真跑就报出：office 的 plan 要 4 个开口，建成几何上只有 3 个；
kitchen 要 5 个，有 3 个。缺口全是窗口，而**分离是精确的**——窗口线段长于 1.0 m 的全部建成，
短于 1.0 m 的全部没有（SPEC-004 §8 的表）。

机制读代码读出来（不是猜的）：Infinigen 的 `make_window_cutter`
（`room/solidifier.py:592`）遍历 `split_mls(mls, self.constants.door_width)`，而
`split_mls`（`:160`）**只吐出比 `min_length` 长的段**。窗口线段短于 `door_width` 时循环体
一次都不执行 → 没有 cutter → **洞口静默地不存在**，建成几何里也没有任何一句说它被要过。
`door_width = (segment_margin − wall_thickness) × uniform(0.7, 0.8)`
（`constants.py:70`）= `(1.4 − 0.1) × [0.7, 0.8] = [0.91, 1.04]`。office 实测门宽 0.9296，
而它在问的窗口线段是 0.926——**差 4 mm**。

这条 gate 的行为是**正确的**：我们给了它一份户型，它没有照建，gate 如实说出来。要动的是
plan 与 provider 之间的那一步。

用户 2026-09-22 的决策：**按 Infinigen 的尺寸，plan 不改，只要开口能有就行。**

## Evidence

- **机制**：上面三处代码（`solidifier.py:160` / `:592`、`constants.py:70`），本轮逐行读过。
  `make_door_cutter`（`:543`）用的是 `max_mls`（**没有长度过滤**），所以**门永远建**——
  静默丢失只发生在窗口上，这是本 ADR 只管窗口的原因。
- **实测（三个实验的 plan）**：窗口线段 1.283 / **0.926** / 1.386（office）、
  1.208 / **0.874** / 1.255 / **0.812**（kitchen）、1.157 / 1.356 / 1.051（bedroom）。
  加粗的是没建成的，全部短于 1.04。
- **抽屉的下界**：`solvers/room.py:1530` 抽的是
  `round(min(rng.uniform(0.8, 1.4), MAX_WINDOW_SPAN_M, usable), 3)`，**下界 0.80 低于
  Infinigen 门宽的上界 1.04**——所以这不是偶发，是两条独立取的常数撞在了一起。
- **抽屉的墙下界**：`_windows_on_wall`（`room.py:1520`）在
  `length − 2 × WINDOW_EDGE_INSET_M < window_width_m` 时一个窗口都不放，即**一个窗口所在的
  墙至少 1.2 + 0.2 = 1.4 m**。这条让"加宽后放不下"在本仓库产出的 plan 上不可能发生。

## Options

1. **抬高抽屉的窗口下限**（`uniform(0.8, 1.4)` → 下界 ≥ 1.05）。改的是 SPEC-002 的
   `_segment_for` / `solvers/room.py`。**否决**：它移动已冻结的房间、改变 `scene_version`、
   三个端测要重跑；更要紧的是**它只修这一支抽屉画出来的窗口**——`RoomPlan.source` 允许
   `external` 与 `bank`，一份外面来的窄窗 plan 照样会静默消失。
2. **把 plan 自己的 `wall_segment_xy` 加宽成宿主跨度**。**否决（用户明确）**：plan 文档变了、
   每个 plan 的哈希与每个 `scene_version` 都跟着动；而且 `width_m` 与线段会变成同一个事实的
   两种写法，二者可以互相矛盾——正是本仓库"一个事实一个出处"要避免的。
3. **选中：plan 一字不动，加宽发生在"翻译进 provider 词汇"的那一步**，
   且**比较与序列化共用同一个函数**。

## Decision

选 **3**。落在两个地方：

```python
# contracts/room_plan.py
INFINIGEN_SEGMENT_MARGIN_M = 1.4          # RoomConstants.global_params(segment_margin=)
INFINIGEN_DOOR_WIDTH_RATIO_MAX = 0.8      # door_params(door_width_ratio=("uniform", 0.7, 0.8))

def buildable_window_span_m(wall_thickness_m) -> float:
    return (INFINIGEN_SEGMENT_MARGIN_M - wall_thickness_m) * INFINIGEN_DOOR_WIDTH_RATIO_MAX

WINDOW_HOST_SPAN_M = 1.10                 # > buildable_window_span_m(0.1) = 1.04，留 6 cm

def opening_host_span(plan, opening) -> segment:
    """门：原样。窗：围绕自己的中心加宽到 WINDOW_HOST_SPAN_M，再夹进它所在的那堵墙。"""
```

**同一条 `opening_host_span` 被两个方向调用**：`floor_plan_payload` 用它写"交给 provider 的
形状"，`contracts/shell_build.py` 的 `_on_segment` 用它判"量到的洞算不算那个计划中的开口"。
两个方向共用一个函数，是这份 ADR 里最要紧的一句——只在序列化那侧加宽，等于拿一个
建造者从没拿到过的跨度去要求它。

**夹不住的时候按名字拒绝，不静默。** 墙比宿主跨度还短时 `floor_plan_payload` 给出整堵墙，
而 `InfinigenShellProvider` 在这个跨度仍短于 `buildable_window_span_m` 时抛
`SHELL_WINDOW_SPAN_UNBUILDABLE(opening_id, span_m, needed_m)`——**这是唯一能说出"这个窗口
为什么没有"的地方**，因为 Infinigen 自己只会把它丢掉。

## Rationale

- **契约早就把这件事说清楚了，本 ADR 只是去花那笔自由。** `PlanOpening` 的
  `width_m` / `height_m` / `sill_z_m` 原文写的是"我们*想要*的，不是对建成几何的承诺"，
  plan 承诺的是"**那个墙上有一个那类的开口**"（SPEC-001 §1、R6）。
  加宽宿主跨度花的正是这笔自由，而且只花在唯一需要它的地方。
- **一个函数，两个方向，不可能互相矛盾。** 这是把"拿一段从没给过的墙去量"这一整类错误
  结构性消掉，而不是用一个够大的容差把它盖住。
- **plan 保持是抽屉说的话。** 加宽在 plan 里做，等于让文档宣称一个没人画过的宽度；在翻译里做，
  plan 仍然是"抽屉要什么"，provider 拿到的是"照着这个建"。
- **失败从静默变成具名。** 这条 ADR 之前，"这个窗口建不出来"这个事实不存在于任何报告里；
  现在它要么被加宽到能建，要么由 `SHELL_WINDOW_SPAN_UNBUILDABLE` 点名说出为什么不能。

## Consequences

**变了的**：

- **SPEC-001 R7 多了第二条主张**：plan 的开口线段必须**落在它宣称的某堵墙上**——
  `check_plan` 新增 `PLAN_OPENING_NOT_ON_A_WALL(opening_id, segment)`。这条以前是隐含的
  （唯一的 plan 来源是抽屉，而抽屉的线段是从它已经知道的墙上算出来的）；
  加宽要夹墙，就必须能找到那堵墙。
- **SPEC-004 的比较集要重述**：逐开口比的是"存在 / `kind` / **在它拿到的那段墙上**"，
  而"那段墙"是宿主跨度。`not_compared` 里 `openings.*.centre_xy` 的理由跟着改：
  从"落在 plan 给的那段里"改为"落在 plan 被翻译成的那段里"。
- SPEC-004 §8 的三个实验表要重记——office 与 kitchen 从 FAIL 变成待验证。

**不变（有意）的**：

- **plan 文档一字不动**：`wall_segment_xy`、`width_m`、`plan_hash`、`scene_version` 都不动，
  SPEC-001 的夹具不用重生成，三个端测不用因为这条而重跑。
- **抽屉的两个常数不动**（`solvers/room.py:1530`、`WINDOW_EDGE_INSET_M`）。
- 门不参与加宽，理由在 Evidence 里（`max_mls` 没有长度过滤）。

**新成本（本 ADR 明确接受）**：

- **比较的沿墙余量变宽了**：一个洞现在可以在 plan 线段之外、宿主跨度之内被接受，
  最多 `(WINDOW_HOST_SPAN_M − width_m) / 2`。这是有界的，且是"这段墙我们确实给过"的那一段；
  垂直方向（在不在那堵墙上）仍是原来那条 0.2 m 的判别，没有放松。
- **仓库里多了一组 Infinigen 常数**（`segment_margin` / `door_width_ratio` 的上界）。
  它们与 `INFINIGEN_PIN` 绑在一起，测试钉住 `WINDOW_HOST_SPAN_M` 覆盖的墙厚区间
  （0.025 m 以上），所以上游改常数时失败的是测试而不是窗口。