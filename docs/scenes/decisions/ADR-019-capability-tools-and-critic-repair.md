# ADR-019：按能力组织工具与只读critic驱动局部修复

状态：accepted design，2026-10-04用户批准；能力是否实现以SPEC/验收为准。

## Context

新Agent已具备静态失败回退与安装修订，但注册混合构建、检索、测量和恢复，
已建实体布局修改不足；视觉critic仅存在于旧生产管线。用户批准先立设计与计划再开发。

## Evidence

定向调研记录SceneSmith commit
5adfd76b1fd43a9a21c0a18983312fc5c7bd613b及工作树差异、SAGE源码摘要与file:line。
SceneSmith把critic作为planner工具，领域designer负责修改；SAGE写操作内部返回critic反馈，
但physics critic有写副作用。二者实际修改/权限/停止边界不能照单继承。
RFW-W有20静态+6机器人阶段项，finish/terminate重复；状态筛选已经存在。
SPEC-040实际已验失败→安装修订→IK/path失效重验；缺陷不等于完全没有迭代。

## Options

1. 增加平铺工具，沿用每个入口各自定义元数据和参数schema。
2. 合并为万能文本编辑工具或立即复制planner/designer/critic多agent拓扑。
3. 一个调度Agent按能力域/操作/scope选择合法动作，内部保留原子事务与独立尺子；
   补受限布局候选工具、通用只读critic与有界修复循环。

## Decision

采用3，分SPEC-041–044独立实现。域为scene/assets/layout/appearance/robot/review/recovery/delivery。
能力目录描述已实现的操作，按stage+name标识（两个finish语义不同），现有工具名/参数保持。
状态合法集合由代码算出；目录分组是呈现和校验，不能越过前置条件。必然依赖允许内部执行，
但工具、子事务、模型请求与墙钟分别记账。第一阶段只组织已有工具，不发布未实现能力。

布局修复输入目标ID/语义约束或工具生成候选ID，连续坐标由确定性求解器产生。
任务身份/目标/硬关系和用户显式锁定字段保持；推断pose允许范围内新版本修订。
支撑搬迁必须保持完整依赖组件，尚不能安全处理时具名拒绝；不能删除子物体来通过。
当前冻结core验收不得被新pose工具绕过；新版本显式绑定上一版本和修订权限，原尺子保留。

critique_scene为只读通用工具：按scope/profile消费静态测量、当前图像和机器人布局证据。
评价过程成功、视觉质量、hard测量通过三者分开。模型提出问题和修复建议，不改场景、不
自报物理/IK成功；未测项not_run。所有证据绑定scene/assets/render版本，修改立即失效。
profile仅在有实际executor时注册；完整动态physics/rollout不会因profile名称被声明通过。

循环由代码保证：当前版本评价→问题→合法候选→实际delta→失效→重测/重评→
finish、限域回退或具名终止。hard义务与配置的必需review未满足时禁止finish。
重复候选、无变化、请求/工具/时间耗尽均终止；不能靠提示词计轮。

## Rationale

分组减少模型接口混杂，原子测量保持可独立验证。范围、版本与只读边界把critic建议变成
可检查的修复请求，避免把“评价完成”误作任务通过，或用删实体/旧图片遮蔽失败。

## Consequences

041组织目录；042实体布局候选与事务；043通用critic；044决策闭环。每能力一个SPEC/PLAN。
旧生产接口与论文支线保持独立；core只用stdlib；渲染/模型/仿真走显式executor。
外发需要PVSCENE_ALLOW_EXTERNAL_VLM=1，模型仅接任务语义/清洗对象ID与授权图像。
当前历史预算27/30，真实VLM验收最多使用剩余3请求并逐请求记账，不能另起隐含预算。
2026-10-04用户后续授权按序小步验证后commit并保留最终端测结果。未来架构重大变化回到ADR/SPEC。
