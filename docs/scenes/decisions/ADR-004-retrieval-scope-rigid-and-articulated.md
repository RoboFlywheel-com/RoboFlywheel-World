# ADR-004：可操作物体检索的目标范围是刚体 + 铰链，软体不做

- 状态：**accepted**（2026-09-21，用户决策）
- 相关：ADR-001（plan 契约）、ADR-002（外壳来源）

## Context

`docs/PRODUCT.md` §5.4 待定"可操作物体检索的边界"。现状：**只有刚体是真的**。
铰链与软体存在于 schema 与 capability 词表里，实现为零，运行时如实报 `unsupported`。

## Evidence

- **能力声明是诚实的**：`configs/capabilities_v1.json` 里 `isaac.rigid` 是 `supported`，
  而 `isaac.articulated` 的 `tested_scope` 写着
  *"no verified articulated asset is present locally; articulated is schema-level only in this
  release"*，`isaac.OPENABLE`/`CLOSABLE` 写着"requires an articulated asset and a joint sweep;
  neither is available in this release"，`isaac.deformable` 同理。
- **词表已经有了但没接线**：`physics_type` 的 `["rigid","articulated","deformable"]` 在
  `schemas/scene-spec.schema.json`、`scene-ir.schema.json`、`asset-record.schema.json` 里都是
  **必填**；`contracts/scene_spec.py`、`assets/capabilities.py`、`orchestrator/escalation.py`、
  `orchestrator/events.py`、`agents/scene_compiler.py` 都引用它们——即"schema/registry/policy 层"。
- **失败阶梯已经声明了不可达**：`orchestrator/escalation.py:165-169` 把
  `ARTICULATION_FAILURE` 与 `DEFORMABLE_SIM_FAILURE` 标为
  *"**Unreachable today**: no articulated or deformable asset exists in the library, so no gate can
  fail on one."*
- **内腔一族完全没有读者**：`AGENTS.md` gap 17 记录 `AssetRecordV3.receptacles` 只被默认化，
  没有任何检索路径构造它，所以每个资产的 `has_receptacle` 都是 `False`，
  `solvers/relation.py::evaluate_inside` 的两个分支**不可能触发**——
  这与 `unsupported_gates` 里的 `capability.INSIDE` 一致。
- 参照项目的做法（`docs/research/comparison.md`）：
  - **SceneSmith**：铰链是它**唯一**做进去的交互类型，表示法是 **SDF + joints=0 的包围盒 +
    Drake FK 调用**（`sdf_mesh_utils.py:116,271`）；`artvip: enabled: true`、
    `partnet_mobility: enabled: false`；有基于 FK 的网格组合验证，
    但**没找到把关节限位扫描当作检索接受条件**的证据。
  - **SAGE**：铰链只有**门**（`RevoluteJoint` + `CreateAxisAttr("Z")` + DriveAPI）；
    物体级铰链只存在于出图脚本，摆放循环里没有关节扫描；
    预期的铰链资产路径**未完成**（`save_layouts_to_usd_w_art.py:25` 导入的 `partnet.urdf_utils`
    在整棵树里不存在）。它的 SPEC 也把 affordance 标为"本期不用"。
  - **Infinigen**：没有资产检索，无铰链库。
  - **软体**：三家都没有可用路径。

## Options

1. 只刚体做深（现状）：铰链/软体只留 schema + `unsupported`。
2. **刚体 + 铰链**（选中）：铰链进库、进 schema，加**关节扫描门槛**。
3. 刚体 + 铰链 + 软体。

## Decision

选 **2**。软体**不做**（只在 schema 保留 `deformable` 词表与 capability 声明）。

铰链从"schema-only"变成真实能力，需要同时满足三件事：
(1) 库里有带关节元数据的资产；(2) 离线做过关节扫描验证；(3) readiness 里有能失败的 gate。

## Rationale

- "围绕**可操作物体**检索"这个产品叙事里，铰链是刚体之外最实质的一类——柜门、抽屉是任务
  本身（开/关/放进去），而不是装饰。只做刚体会让产品定义里的"可操作"缩水成"可抓取"。
- 参照项目证明这条路的**表示法已经成熟**：SceneSmith 的 SDF + joints=0 bbox + FK 是可直接
  复用的形状，且它已经有 ArtVIP 这条验过的来源。
- 软体不做，是因为**没有任何参照项目有可用路径**，而且它会引入独立的时间尺度与接触模型，
  与"readiness 阶梯"的现有语义冲突。

## Consequences

**必须先确认的前提（不是既成事实）**：

- **当前 SimReady 库（v1.3）是否含铰链资产，尚未核实。** 库的 SPEC（v0.2 口径）写的是
  "**单 link 刚体**"，`part_physics.json` 是逐 part 物理属性——这与"铰链"不是一回事。
  本 ADR 的**第一个待办**是先核库；若库里没有，则铰链范围需要一条**资产来源**决定
  （参照：SceneSmith 用 ArtVIP；PartNet-Mobility 在它那里被**默认禁用**，理由是已知质量低）。

**要做进 readiness 的**：

1. **新 gate**：关节扫描无碰撞、静置稳定；`unsupported_gates` 里的
   `capability.articulated.*` 只有在**有已验证资产且扫描通过**之后才允许翻成 supported
   ——`assets/capabilities.py` 已经把"目录声明"与"后端验证"分开，这个纪律必须继续。
2. **摩擦端到端**：SAGE 的教训是**摩擦被硬编码**（`1e20` / `2.0`），逐资产的 `mu` 永远到不了
   USD，于是它所谓的 sim-ready 只有质量 + 碰撞壳。**我们的库有 `mu1`/`mu2`**，
   必须携带进交付物并**验证它真的落进了 USD**——"写进去"与"验到"是两件事。
3. **验收差额清单**（本 ADR 一并接受，来源是 SAGE 自己的失败语料，
   它的物理 critic 一条都没检出）：把下面五条写成 gate，而不是指望生成过程不出错——
   (a) 除按类型本就落地者外，不许有物体底面在地面高度；
   (b) 支撑关系必须几何为真（底面包含于支撑顶面、底面高度在容差内）；
   (c) 家具与其座具的高度一致性；(d) 逐物体尺寸上界来自**类别**而不是模型估计 + 随机收缩；
   (e) 岛台/通道类物体的净空余量。
   另加：**整场景**（而非逐房间）验证。

**被推迟的**：

- 软体；`capability.INSIDE`（内腔）仍然 `unsupported`，因为 "认证内腔" 需要先有构造它的
  检索路径（见上文的 receptacle 空转问题）——它是**另一个**目标，不是铰链的附带产物。

**未决**：

- 铰链资产的来源（自建索引 / ArtVIP / 其它）——先核 v1.3，再定。
- 关节元数据的存放位置（沿用 `part_physics.json` 的逐 part 结构，还是新增 sidecar）。