# ADR-002：米制房间外壳由 Infinigen provider 生成

- 状态：accepted（2026-09-21）；实现布局按公开包重构更新。
- 相关：[ADR-001](ADR-001-room-plan-as-the-core-to-room-contract.md)、[ADR-007](ADR-007-generated-background-around-frozen-core.md)。

## Context

内置外壳用于几何预览；完整建筑需要门窗、踢脚及程序化材质，同时须服从已确定的任务空间。
随机房间尺寸不能满足米制 RoomPlan，也不能通过整体缩放修复：缩放会改变门高、家具及碰撞。

## Decision

通过 `rfw_w.scenes.solvers.shell_providers` 接入独立 Infinigen 源码，固定 commit
`892947be71c4ff102c6a558d088ffe149561c11f`，本机位置由 `RFW_INFINIGEN_ROOT` 提供。
该版本提供 `PredefinedFloorPlanSolver`；缺少预定义户型能力的版本不能替代。
正式入口是安装后的 `rfw_w.scenes.workers.blender.run_infinigen_build`，运行补丁也由本包管理。
上游源码保持原样，入口/补丁摘要及 pin 写入构建记录。移植代码的许可见根 THIRD_PARTY.md。

户型以米制多边形、墙上门窗线段定义，调用预设路径；结构与家具生成 stage 分开控制。
不直接接受上游缺失的语义序列化作为证据。独立测量建成网格，验证 RoomPlan 尺寸与开口。
模型生成字符串不能作为上游 eval 输入；户型表达式只能由确定性 writer 构造。

## Consequences

- 几何测量与房间契约不因 provider 改变；外壳测量失败必须阻塞相应交付等级。
- 材质是程序化 shader。贴图烘焙是独立 Blender/Cycles 工作，GPU能力逐次检查。
- 门窗与建筑细部 stage 显式配置；不能从旧运行结果推断当前生成成功。
- 生成耗时与房型有关，离线房间银行仍是可选方向；当前不声称已有完整银行。
- 房间及外围的运动、可开合和任务认证不由建筑导出自动获得。
- 预设户型若无法稳定匹配测量，重新评估内置几何或独立材质路径。
