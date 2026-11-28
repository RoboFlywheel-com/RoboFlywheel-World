# ADR-003：机器人输入由 URDF 提供，取尺寸包络，不承诺运动学

- 状态：**accepted**（2026-09-21，用户决策）

## Context

`docs/PRODUCT.md` §2 要求输入含**机器人型号与 URDF**。现状是：机器人只由**一个字符串名字**
标识（`TaskSpec.robot_profile`，必填非空），解析到一张**只有一条记录**的硬编码表。

## Evidence

- `geometry/robot.py:35` `RobotProfile` 就是**四个数**：`name` / `footprint_m` / `reach_m` /
  `approach_clearance_m`，`base_radius_m = footprint_m / 2`。它的 docstring 主动划了边界：
  *"Deliberately four numbers and not a kinematic model. A full model would let a gate lie with more
  confidence… It cannot say a grasp exists, and it must not pretend to."*
- `ROBOT_PROFILES`（`:74`）**只有一条**：`franka_panda`（0.6 / 0.855 / 0.15）；
  `profile_for()`（`:80`）对未知名字**报错**而不是替换成别的臂。
- **全仓没有任何 URDF 摄取、没有运动学链、没有 IK 求解器**：对 `urdf|kinematics|ik_solver|
  import mujoco|pinocchio` 的全仓检索只命中"资产元数据 sidecar 的路径名"与契约测试里的字符串；
  `robot.nav` / `robot.ik` / `robot.motion` / `task.rollout` 一律 `not_run`，
  因而 `robot_feasible_scene_ready` 与 `task_verified` **由构造不可达**。
- 机器人数据已经在多处被消费，且**有一处已经不一致**：
  - `solvers/cell.py:255` `profile_for(spec.task.robot_profile)`，protected 区域用
    `-robot.reach_m - robot.base_radius_m`；
  - `solvers/room.py:1686` `check_door_connectivity` 用 `robot_footprint_m + clearance_margin_m`，
    而 `ShellOptions.robot_footprint_m`（`:291`）是**默认 0.6**，不是从 profile 读的；
    `solvers/zones.py:503/:552` 同理（`corridor_width`）；
  - `rfw_w.scenes.commands.generate:854-869` `_space_conditions` **从 `generate_options(spec)` 取
    `robot_m`，而不是从 `RobotProfile`** —— 同一个机器人有两个来源；
  - `run_m1_acceptance.py:1030` `_task_reachable` 是 reach **proxy**，
    gate 的 `method` 字段自己写着 `reach_proxy_standoff_solve`、detail 写着 "not a grasp"。
- 参照项目的结论（`docs/research/comparison.md` §2.2）：**三个参照都没有把机器人 URDF 当输入**。
  SceneSmith 用一个 `robot_width: 0.3`；SAGE 的机器人在 Isaac 里是 **USD 引用**
  （franka/jetbot/carter/G1/Go1/ridgeback_franka），**仓库里根本没有机器人 URDF**。
- `AGENTS.md` gap 16 记录了换型号真正的风险：`standoffs_for` 描述的是**移动机械臂**
  （基座走到目标旁），而 `piper_x` 是**固定在桌面上的底座**——对后者，"站位带"这个问题本身
  就不成立，**只换 profile 会让 reach gate 声称基座能站在一条并不存在的过道里**。

## Options

1. **URDF 取尺寸包络**（选中）：footprint / reach / 高度包络来自 URDF，`nav/ik/motion` 保持 `not_run`。
2. 选项 1 + FK：再解析运动学链做前向运动学。
3. 选项 1 + IK / 可达性判定：可达性从 proxy 变真判定，改 `robot_feasible_scene_ready` 的语义与门槛。
4. 不做 URDF，继续名字 + 硬编码表。

## Decision

选 **1**。URDF 是**输入**，但当前只被用来派生**尺寸包络**：footprint（或基座半径）、reach、
高度包络，加上一个必须显式声明的**基座类型**（移动 / 固定）。

## Rationale

- 产品要求 URDF 作为输入，而尺寸包络是**唯一在当前架构里既真实有用、又不需要仿真器**的部分：
  它已经驱动 protected 区域（`cell.py`）、门宽 gate（`room.py`）、走廊宽度（`zones.py`）
  与 reach proxy（`run_m1_acceptance.py`）。把它从"硬编码表"换成"URDF 派生"是净收益。
- 不选 2/3 是因为**三个参照项目都没有做**，而 `robot.ik` 的真判定会改变
  `robot_feasible_scene_ready` 的语义——那是另一个里程碑的决定，不该在"输入格式"这一层顺手改掉。
- 保留 `RobotProfile` 的四数形状是有意的：它现在会诚实地说"这不是抓取"，
  换成完整运动学模型只会让 gate 更有信心地说错话。

## Consequences

**必须做的**：

1. **解析在离线/资产准备期完成**，结果落成一条带来源的 `RobotProfile`；在线路径只读表。
2. **`RobotProfile` 增加来源字段**（URDF 路径 + 内容哈希），使"这个数是量出来的还是抄来的"
   可追溯——与 `AssetRecordV3` 的 `prepared_version` / `source_hash` 同一纪律。
3. **修掉已经存在的双来源**：`ShellOptions.robot_footprint_m` 与 `_space_conditions` 的
   `robot_m` 必须改成从 profile 读，否则 URDF 换了而门宽 gate 还用 0.6，是**静默的假通过**。
4. **基座类型进 profile**：`mobile_base` / `fixed_base` 是一等字段，因为 `standoffs_for` 的
   "站位带"语义只对前者成立（gap 16）。固定底座的型号必须走另一套 gate 语义，
   而不是复用站位带。**这是本 ADR 最重要的一条，也是最容易被漏掉的一条。**

**多型号**：

- 架构上**表驱动**，且每条 gate 要**逐型号分别成立**——不能把某一型号的假设内联进 gate；
- 实现上先一条。`profile_for()` 对未知名字报错的行为保持不变（不回退到别的臂）。

**未决**：

- 高度包络在 v3 的 gate 里目前没有消费者（`_space_conditions` 只用 footprint）；
  引入它需要有 gate 真的读它，否则就是"解析了但没人用"的字段（`AGENTS.md` gap 17 的同类问题）。
- 若将来做选项 3，多型号成本显著上升，届时应重新开一条 ADR。