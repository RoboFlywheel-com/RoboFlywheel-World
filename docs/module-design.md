# RFW-W 模块组织

RFW-W 使用一个根级 `src/`、一个安装包和一份 `pyproject.toml`。
`rfw_w.assets` 与 `rfw_w.scenes` 分别承载两种用例，公共命令负责独立进程执行。
依赖按功能 extras 安装；各 worker 可在不同解释器安装同一个包。删除的生产目录及入口不提供兼容调用。

```text
src/rfw_w/                        公共命令、运行环境隔离
src/rfw_w/assets/
  application.py                 单资产生成用例
  contracts/                     请求、产物、物理记录
  geometry/                      质量检查、分割、碰撞
  annotation/                    语义、材料及部件标注
  physics/                       密度、质量、COM、惯量
  packaging/                     独立资产包
  catalog/                       索引、检索及浏览资源
  simulation/                    格式导出、后端与任务验证
  workers/                       模型环境中的工作进程
src/rfw_w/scenes/
  application.py                 场景生成与有界修复
  assembly/                      摆放、几何、相机、灯光
  contracts/                     SceneSpec、SceneIR、证据
  assets/                        检索、测量、资产准备
  geometry/                      几何与碰撞计算
  solvers/                       任务、房间、关系求解
  orchestrator/                  图执行、工具、回滚
  delivery/                      交付与证据组合
  validation/                    独立重验与负例审计
  workers/                       引擎、Blender、任务执行
  preparation/                   索引、权重及数据准备
  visualization/                 可选渲染与展示
  resources/                     安装随附的配置、schema、小型输入
```

## 单资产生成

一份请求明确选择 prompt、image 或 mesh 起点，包含生产身份、seed 和所需格式。
批量调度复用单资产用例。模型、解释器和本机路径在独立运行配置中提供。

生成 → 质量检查 → 分割与标注 → 尺度/坐标准备 → 碰撞与体积 → 物理 → 资产包。
确定的物理记录供导出器使用，避免装配时再次估计参数。质量校准同步更新惯量。
资源引用留在包内；包可复制到新目录进行独立验证。

生产 object_id 与发布 leaf 分开；v1.3 发布身份通过显式 mapping 关联。
生成本地包不更新全库、不上传。现有发布库只读，物理以 `part_physics.json` 为来源。

## 场景生成

SceneSpec → 资产检索/准备 → 核心摆放 → 房间 → 外围 → 修复 → SceneIR 与 delivery。
Agent 负责角色、任务语义和关系；具体绑定、尺寸、位姿、碰撞由执行模块完成。
保留任务初态与目标态分离、保护集合、坐标准备记录、不可变回滚及证据失效规则。

单资产包可通过显式生成绑定直接作为任务目标；数据由场景 provider 独立读取，
不导入资产生产包。包原始坐标、物理与碰撞分解保留，当前单mesh worker使用凸代理
并推导COM/惯量，其实现等级单独留证据。直接绑定支撑体与上下文仍未实现。

每次运行使用独立状态记录拒绝和缓存，安装包通过资源接口加载默认配置。
正式模块直接导入所属实现，不从开发脚本路径动态加载。

## 仿真交付与验证

目标后端为 Isaac Sim、MuJoCo、Genesis、PyBullet、SAPIEN；引擎依赖在各自 worker
环境中加载。资产导出与场景渲染目前各有具体实现；共同 world 协议和完整物理验证
仍需收敛，不能将模块迁移当作跨仿真器验收完成。

每后端分别报告 export、load、render、step、settle、relocate_verify、task_run。
渲染通过不代表稳定或任务成功；缺依赖和未执行项目明确报告。
独立验证重读文件和摘要，使用新目录、新进程，不继承生产者的成功声明。

## 安装与检查

安装根级可编辑包或单个 wheel 后，使用 `rfw-w asset …` 与 `rfw-w scene …`。
从仓库之外 cwd 检查命令、配置/schema和 worker 的可达性。
核心保持标准库可导入；科学计算、模型、引擎和浏览功能通过可选依赖安装。
当前入口见两侧 README；验收范围见[公开复现规格](public-reproduction-spec.md)。
