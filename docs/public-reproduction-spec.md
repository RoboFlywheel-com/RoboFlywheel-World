# RFW-W 复现范围

RFW-W 包含单资产生成与任务场景生成。安装与运行见 [复现指南](reproduction.md)，
模块边界见 [模块设计](module-design.md)。

## 完成标准

- 安装包从仓库外运行两条管线；记录输入、seed、模型版本、配置和产物摘要。
- 请求资源先预检，失败保留阶段与原因；资产包使用相对资源引用，坐标准备可追踪。
- 生成与独立迁移验证使用新目录。已有发布库仅作只读对照，生产身份与 leaf 经映射解析。
- Isaac Sim、MuJoCo、Genesis、PyBullet、SAPIEN 逐项记录 export、load、render、step、
  settle、relocate_verify、task_run；未配置资源记录 blocked/not_run。
- 机器人抓放、连续轨迹、跨房任务和模型物理估计精度单独验收。
- 公开源码、文档和配置模板不包含凭证、私有端点或个人路径。

## 当前覆盖

| 能力 | 结果 |
| --- | --- |
| prompt 起点七阶段及 URDF/MJCF/USD 导出 | passed |
| 合成与真实 mesh 生成、复制后验证 | passed |
| 四后端资产加载、物理参数与步进 | passed |
| 生成资产绑定两实体场景与独立交付检查 | passed |
| 场景请求的 MuJoCo 凸代理步进 | passed |
| 原生 SBERT 部分类别索引、检索与准备 | passed；完整库未覆盖 |
| Isaac、完整五后端场景与动态机器人任务 | not_run / blocked |

场景原生多 hull、完整惯量以及支撑体/上下文直接绑定仍需扩展。
资产目录检索与场景语义检索保持独立后端；渲染、加载与行为分别提供证据。
