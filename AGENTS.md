# RFW-W 工作约定

RFW-W 统一承载刚体资产生成与任务场景生成。

- 模块与接口变更先读 [模块设计](docs/module-design.md)；场景变更再读
  [场景架构](docs/scenes/ARCHITECTURE.md)。领域用语见 [GLOSSARY.md](GLOSSARY.md)。
- 源码统一在 `src/rfw_w/assets` 与 `src/rfw_w/scenes`；单资产与批量共享实现。
  各 worker 环境安装同一包，重依赖在执行阶段加载。
- 本机配置走环境变量或 Git 忽略文件。权重、第三方源码、完整库与运行产物存放在仓库外；
  新产物目录同步 `.gitignore`，根 `assets/` 仅存项目配图。
  凭证只检查 set/unset；源码、文档和提交中不得包含凭证、私有端点或个人路径。
- 模型只接收授权的任务语义或图像，不发送仓库文件、路径或日志。
- 外部发布库只读，新运行使用新目录。物理参数来自 sidecar，摆放采用实测几何，
  坐标转换保留记录；生产身份与发布 leaf 通过映射关联。
- Agent 描述语义与关系；执行模块负责资产绑定、位姿、碰撞和仿真。
- 验证入口为 `bash scripts/check.sh`，核心导入检查加 `--stdlib-only`。
  普通 Python 清除 Isaac 注入的 `PYTHONPATH`、`PYTHONHOME`；模块变更运行相关契约检查，真实资产接入增加 integration。
- 分别报告 passed、skipped、not_run、blocked。渲染、加载和任务执行逐项提供实际证据。
- 提交与推送按用户授权执行；仅提交当前任务的文件。完成时说明验证结果和剩余限制。

安装与运行见 [复现指南](docs/reproduction.md)，公开文件边界见 [security.md](docs/security.md)。
