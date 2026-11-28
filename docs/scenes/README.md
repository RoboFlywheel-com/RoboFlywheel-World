# RFW-W 场景生成

`rfw_w.scenes` 将任务规格、配置资产和几何约束转为 SceneIR、仿真请求与交付证据。
安装根级包后可从任意目录运行；安装步骤与已测版本约束见 [环境与复现](../reproduction.md)，
求解顺序与契约见 [场景架构](ARCHITECTURE.md)。

## 运行

```bash
rfw-w scene generate --experiment office_pick_cup --compile-only --output-dir outputs/preview-001
rfw-w scene generate --spec /path/to/task.json --env-file /path/to/local.env --output-dir outputs/scene-001
rfw-w scene validate --delivery outputs/scene-001
```

预览不调用模型、检索或仿真。输出使用新目录，运行分别记录检索、摆放、修复与证据。
生成退出码依据所请求 readiness 和后端检查；缺少请求的 load/run/render 会返回 1 并保留产物。
独立验证检查声明与证据的一致性，验证通过不代表未运行的仿真已通过。

自然语言输入用 `--prompt` / `--prompt-file` 及编译模型配置。
服务接口由使用者配置，外部模型需要 `PVSCENE_ALLOW_EXTERNAL_VLM=1`；
模型描述语义、物体角色与关系，执行模块负责绑定、位姿和碰撞。

## 单资产接入

资产包通过 `rfw.asset-package/v1` 文件协议衔接，包须包含 URDF。
binding JSON 将任务目标映射到包目录，例如 `{"block":"../asset-001/package"}`；
相对路径以 JSON 所在目录为准。资产许可证独立声明，未记录时使用 `unrecorded`。

```bash
rfw-w scene generate --experiment synthetic_asset --asset-bindings /path/to/bindings.json \
  --asset-license unrecorded --output-dir outputs/bound-scene-001 --request-only --no-replan
```

内置两实体输入使用目标与构造支撑体，无需发布全库、SBERT、HSSD 或模型。
当前直接绑定支持单个任务目标；支撑体或上下文绑定报 `GENERATED_BINDING_UNSUPPORTED`。
未绑定目标继续检索，额外家具与桌面内容需要配置相应资产来源。

接入核对 `_SUCCESS`、文件摘要、URDF 引用、物理 sidecar、坐标记录与实测尺寸，
将资产独立复制到 `prepared_assets/generated/`，保留原始尺度及碰撞 hull。
当前场景单 mesh 请求合并为凸代理，质量进入 worker，COM/惯量由引擎推导；
逐实例记录 `collision_realization`、`physics_realization` 和原始 `rigid_physics`。
多 hull 与完整惯量传递、动态任务仍需验收。

`--request-only` 生成摆放与请求。静态交付可用 `--skip-isaac --no-loop --no-replan`，
缺后端验收时返回 1，独立验证仍可检查其声明一致性。

## 配置与检索

使用 [runtime.env.example](../../examples/scenes/runtime.env.example)，本机配置由 Git 忽略。
worker 用 `RFW_SCENE_PYTHON` 或公共命令 `--python` 选择，各环境安装同一包。
以下是现有配置键，接口兼容保留其拼写：

| 设置 | 用途 |
| --- | --- |
| PVSCENE_SIMREADY_INDEX / PVSCENE_SIMREADY_ASSETS_ROOT | v1.3 目录索引与 assets 根 |
| PVSCENE_SIMREADY_SEMANTIC_INDEX | 本地 SBERT 类别索引 |
| PVSCENE_PREPARED_DIR | 权重与资产准备根 |
| PVSCENE_HSSD_ROOT / PVSCENE_HSSD_SYNSET_INDEX | 可选家具来源 |
| RFW_INFINIGEN_ROOT | 可选 Infinigen 固定版本源码 |

```bash
rfw-w scene retrieve --entity /path/to/entity.json --admit Mug 'Coffee Cup' --out outputs/retrieve-001 --prepare
python -m rfw_w.scenes.preparation.build_simready_index --help
python -m rfw_w.scenes.preparation.build_semantic_index --help
```

semantic 模式需要本地编码器和类别索引，失败不静默转 lexical；显式 lexical 只检查目录与几何。
权重 revision、模型身份、向量维度和发布版本须匹配。
场景目录/SBERT 与资产 FTS5/bge-m3 尚未统一后端和索引契约。
实际全库检索、三种场景与改进建议见 [检索检查](retrieval-evaluation.md)。

## 验证

```bash
bash scripts/dev/test-scenes.sh
bash scripts/dev/test-scenes.sh --stdlib-only
bash scripts/dev/test-scenes.sh --assets --require-assets
bash scripts/dev/test-scenes.sh --delivery
```

从仓库根运行；测试解释器由 `PVSCENE_RUNTIME_PYTHON` / `PVSCENE_STDLIB_PYTHON` 选择。
普通 Python 清除 Isaac 注入的 `PYTHONPATH` / `PYTHONHOME`，缺资源明确失败或报告未运行。
五后端 worker 与原生任务模块分别记录证据；完整跨后端场景与机器人任务尚未验收。
