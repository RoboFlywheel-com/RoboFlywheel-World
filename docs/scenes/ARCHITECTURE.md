# 场景生成架构

正式场景生成用例位于 `rfw_w.scenes.application`；命令由 `rfw_w.scenes.commands.generate`
解析参数。模块从安装资源加载配置与小型示例，worker 在各自解释器环境中运行。

## 执行顺序

CORE → ROOM → PERIPHERY：先解任务核心，再围绕保护空间生成房间，最后添加外围物体。
单房求解与任务 world 图执行分别保留现有 solver/orchestrator。

| 模块 | 职责 |
| --- | --- |
| contracts | SceneSpec、SceneIR、资产、图、证据及单位；标准库 |
| application | 生成请求、运行状态、有界修复、预算与退出结果 |
| assembly | 摆放、几何检查、生成家具、相机、灯光 |
| assets | 目录检索、真实几何测量、资产准备与绑定 |
| geometry / solvers | 多边形、净空、支撑、关系和布局构造 |
| orchestrator | 工具执行、checkpoint、不可变回滚、证据失效 |
| delivery | manifest、准备度解析及运行证据组合 |
| validation | 从交付文件独立重验；负例与环境检查 |
| workers | Isaac、MuJoCo、多后端渲染、Blender、可选任务执行 |
| preparation | 本地索引与权重准备 |
| visualization | 可选可视化与媒体生成 |
| resources | 安装随附配置、schema、示例 |

## 稳定契约

- SceneSpec 区分生成时 initial_relations 与动作后的 goal_predicates。
- SceneIR 记录房间多边形、实例、资产绑定、支撑图、任务关系、坐标准备及内容摘要。
- 支撑图无环；任务关系可以成环并作为约束组求解。
- CORE、任务关系闭包、hard_required 是不同集合，保护与拒绝规则不能互相替代。
- m/kg/s/rad、右手 Z-up、列向量 T_parent_child、JSON quaternion wxyz。
- USD 的行向量约定显式转换；已经准备的几何不再居中、旋转或缩放。
- 每个运行具有独立拒绝记录和缓存；移动实例及回滚使关联证据失效。
- Affordance 状态区分 measured/annotated/estimated/unknown；未知不是已验证成功。
- 资产 identity 通过版本化目录/mapping 解析，尺寸由几何测量，物理追踪 sidecar。

## 模型与后端

编译任务、视觉 critic、replan 通过明确授权的模型 adapter 执行。
本机配置提供模型、端点、凭证变量名与预算，预检不触发模型调用。

内置房间 shell 不需要外部项目。可选 Infinigen 使用独立、固定版本的 checkout；
本仓库负责生成入口、约束注入、导出与测量，安装到 Blender Python 后运行。
引擎模块不修改任务语义，也不隐式改变资产尺度和质量。

五种引擎的渲染代码已作为正式 worker 组织。完整五后端 load/step/settle/relocate 验收
仍需实际运行；静态 RGB/depth 不能证明动态物理或机器人任务成功。

## 独立验收

交付 manifest 明确记录 requested/achieved、not_run、unsupported 与实际证据。
独立 verifier 重读产物、摘要、图版本和准备度解析，不相信生成者的 passed 声明。
实际负例和契约测试随模块迁移，安装与外部 cwd 测试保证能力不依赖源码目录。
相关决策保存在 `docs/decisions/`。
