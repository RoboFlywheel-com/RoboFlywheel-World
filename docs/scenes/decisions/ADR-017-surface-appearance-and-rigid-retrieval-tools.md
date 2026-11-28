# ADR-017：墙地外观、覆盖物与刚体检索工具的边界

日期：2026-10-03。状态：设计决定；实现与验收待 SPEC-028–030。

## Context

用户要求新两房 Agent 可选择墙地风格、增加墙地内容，并使用 SimReady 可操作刚体。
现有预览硬编码浅色，墙面仅薄面板资产，G3 资产链只接受 HSSD。

## Evidence

定向调研 含固定参照 commit、
逐项 file:line 与已提交普查工具。SceneSmith 材料选择在 floor-plan 阶段，
墙面物件以建成墙面局部坐标放置，thin covering 独立于刚体。
本机有两套 PBR 与配置 SimReady v1.3 目录；已有 adapter 可复用。

## Options

1. 只改预览颜色、给阶段包装器增加 SimReady 名称：无法追溯材质，也无法通过现有 HSSD 验证器。
2. 复制参照的房间重建、资产生成和阶段 Agent：扩大边界，且引入未经验证的外部服务。
3. 独立材料/覆盖物/刚体工具，在统一图中保存意图与产物引用，沿用事务、哈希、回退和独立验证。

## Decision

选择 3。按 SPEC-028、029、030 独立验收，完成各自工具后接入现有 AdaptiveSceneAgent。

- 面外观记录在 `built.surface_appearance`，里面区分用户指定/推断的样式意图和
  已准备的绑定；外部材料账只是图引用的产物，禁止第二份可编辑意图。
  第一版支持按房间 floor / wall_set，以及实际墙段的 room-side 覆盖。
- 共享墙两侧是两个独立面绑定；不以整个共享 OBJ group 单一材质覆盖两房。
  实际网格、核心实例不变；外观改变使旧视觉证据失效，新图仍需重新关联/核验几何账。
  不能把旧版本验证直接复制成新版本通过。
- 地毯、墙纸、海报使用显式 thin-covering 类型，挂在 floor 或 wall_set 下，
  有建成面 anchor、边界、法向、覆盖材料与独立实例 ID。
  composition/v3 显式迁移，只扩 `physical_class=covering`，不改变现有 placement_kind。
  首例地毯为无碰撞视觉覆盖，墙面用几何间隙代理检查，物理功能均不声称成立。
- HSSD 提供家具/合适的背景；SimReady 首例提供 surface manipuland，provider 选择显式，
  不自动退回 HSSD。SimReady search 是只读候选工具，select/prepare/assemble 是写工具。
  同一进程复用 SBERT engine，不建新常驻服务。
- 检索先用 provider 的明确准入类别限制候选，再排序，再真实网格测量；
  类别名与全库目录不发送给在线模型，只给当前角色的有限候选摘要。
  资产键为 provider/version/id，准备路径有 provider 命名空间。
- 第一版桥接准备产物至底面为零、XY 居中的 frame；visual 与 collision 使用同一变换。
  质量/摩擦为目录声明并有来源标记，不因此认定可抓取或 settle 成功。
- 新写工具均使用 expected_hash、fresh output、touched_ids 与旧证据失效。
  覆盖物/外观回退局部重验；核心目标换 provider 使后续装配和机器人义务全部失效。

## Rationale

这些能力有不同失败条件，能够独立测试和审查。几何不必因颜色变化重建，
但显示与语义证据必须随外观版本重测。刚体来源由可测资产决定，模型只选择候选。

## Consequences

需要对 G3 的 HSSD 特定校验做 provider 分派，对 v3 composition 显式迁移，
并扩动态 Agent 的合法工具参数；不能靠 patch_index 内置方案冒充开放材料/资产检索。
首版本地 PBR 只有两套，丰富材料目录/CLIP 服务是后续能力，不记为已实现。
新增 SPEC 不包括机器人、物理地毯、灯光电气、可承重墙架或在线资产生成。
