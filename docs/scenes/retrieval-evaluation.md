# RFW-W 场景检索与生成检查

2026-10-29 使用 v1.3 目录索引的 524,564 个资产、1,212 个类别，重新构建本地 SBERT
类别索引。权重为 all-mpnet-base-v2，固定 revision 见 [复现指南](../reproduction.md)。
已有发布库只读，所有准备和运行产物写入新的外部目录；未调用远程模型。

## 当前实现

任务目标经 SBERT 类别召回，再按物理类型、尺寸和文件准备度排序；场景 adapter 执行
明确的类别准入、真实 mesh 测量与坐标准备。无候选返回 `ASSET_UNAVAILABLE`，不会把
邻近类别作为成功替代。源码见 [retrieval_engine.py](../../src/rfw_w/scenes/assets/retrieval_engine.py)
与 [simready_retrieval.py](../../src/rfw_w/scenes/adapters/simready_retrieval.py)。

家具及桌面上下文采用 HSSD synset 类别索引、名称/材料提示与几何尺寸筛选。
支撑体按任务核心构造；生成资产包也可明确绑定为任务目标。
资产浏览侧 FTS5/bge-m3/RRF 仍为另一套实现，尚未共用后端与索引契约。

## 实际检查

| 输入 | 检索 | 场景请求 | MuJoCo 检查 |
| --- | --- | --- | --- |
| office_pick_cup | 8 个准入候选；首选实测尺寸符合声明区间 | 12 个实例 | 600 步 passed |
| kitchen_fetch_jar | 8 个准入候选；首选实测尺寸符合声明区间 | 9 个实例 | 600 步 passed |
| bedroom_pick_alarm_clock | 8 个准入候选；首选实测尺寸符合声明区间 | 9 个实例 | 600 步 passed |
| 缺失 Desk 类别 | `ASSET_UNAVAILABLE` | 未生成 | not_run |

三种场景均从真实检索结果生成，记录 SceneIR、摆放与仿真请求；未绑定合成占位资产。
MuJoCo 检查涵盖请求中的所有实例、实际质量与有限状态步进，使用单 mesh 凸代理、引擎估计
惯量和地面。未检查墙体碰撞、settle、RGB、机器人操作或跨后端场景。
该样本验证生成路径，不构成全库检索质量或物理精度评测。

办公室额外执行无修复循环的静态交付检查：`geometry.no_penetration`、支撑和任务关系通过，
`geometry.in_bounds` 与 `function.space_conditions` 失败。5 个外围实例越过墙内边界约
0.05–0.11 m，家具切断入口到任务区的可通行区域。交付明确为 `draft` / `meets_request=false`；
独立 verifier 复核一致，不能将其通过解释为场景达到功能准备度。
默认修复循环能否解决这两类失败尚未端测，完整后端/机器人验收仍为 not_run。

默认检查：资产与公共入口 64 项 passed；场景 unit/contract 2,485 项 passed、3 skipped、
6 deselected；真实检索与 HSSD 支撑面 integration 21 项 passed（禁止跳过）。
标准库检查 146 个核心模块通过，内置敏感扫描零命中；本机关键词扩展未配置。
在已测约束外的新 trimesh/PyOpenGL 版本曾触发材质/EGL 回归，采用
[geometry.constraints.txt](../../examples/reproduction/geometry.constraints.txt) 后相关检查通过。

全库加载还暴露了发布身份校验缺陷：5,336 个有效资产的序号超过 9999，原来的四位上限
导致整个目录被拒绝。资产目录构建、路径解析和场景目录校验现已接受至少四位的编号，
保留前缀一致性与路径约束；回归测试覆盖四位和五位身份。

## 判断与下一步

现有方案适合作为可复现基线：类别准入限制语义替换，实测几何决定摆放，失败有原因。
完整公开方案还需补齐以下能力：

- **统一索引与入口**：资产生成产物、发布库与场景调用共享版本、身份、物理和许可契约。
  保留类别、尺寸和任务约束准入，再组合关键词与语义召回。
- **任务与实例信息**：类别向量不足以区分把手、开口、容量或具体材质；当前 affordance
  排序值为常量，不能作为能力证据。补充逐资产标注，并用独立查询集评估准入后 recall/top-k。
- **物理忠实度**：当前检索路径重新构造单凸代理，并采用零 COM 和箱体惯量。
  下一步让 scene provider 保留 sidecar 物理、原始 hull 与坐标变换记录，避免准备过程降低资产精度。
- **布局验收**：请求生成前也应运行墙内边界与通路检查，并在最终物体姿态/尺寸改变后
  重新计算净空。优先修复办公室已测到的外围越界和入口断路，再验收修复循环。
- **检索成本**：类别准入在 mesh 测量之后，导致读取随后被拒绝的资产；本轮三个查询各耗时
  约 65–114 秒，包含挂载盘 I/O。把准入前移、持久化几何与准备度元数据，只准备最终候选。
  家具筛选也应减少挂载盘逐文件 stat/read。

安装与复现命令见 [场景指南](README.md) 和 [复现指南](../reproduction.md)。
