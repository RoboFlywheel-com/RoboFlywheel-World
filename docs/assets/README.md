# RFW-W 资产生成

`rfw_w.assets` 将 prompt、image 或 mesh 请求生成独立资产包。单资产与批量共享实现；
模型、物理计算和仿真器通过 runtime 指定的独立解释器运行。
安装步骤见 [环境与复现](../reproduction.md)，接口见 [模块设计](../module-design.md)。

| 模块 | 职责 |
| --- | --- |
| contracts / application / batch | 请求验证、阶段编排、证据与批量调度 |
| generation / workers | 图像、去背、三维生成、分割、渲染与标注 |
| geometry / physics / annotation | 几何质量、尺度、碰撞分解、部件属性、质量与惯量 |
| packaging | 包内相对引用及 URDF/MJCF/USD 导出 |
| catalog | v1.3 目录、FTS5、bge-m3/RRF 与浏览服务 |
| simulation / visualization | 五后端适配、独立检查与展示 |

## 请求与运行

[请求示例](../../examples/assets/request.json)选择一个输入起点；
[runtime 模板](../../examples/assets/runtime.example.json)按阶段配置解释器、资源与服务接口。
`options` 对应 worker 参数，`env` 仅作用于该进程；依赖与资源先预检，缺配置明确失败。

```bash
rfw-w asset generate --request /path/to/request.json --config /path/to/runtime.json --out outputs/asset-001
python -m rfw_w.assets.commands.batch --request /path/to/requests.jsonl --config /path/to/runtime.json --out outputs/batch-001
rfw-w asset validate --bundle outputs/asset-001/package --out outputs/check-001 --backend mujoco --steps 600
```

输出目录必须不存在。mesh 输入可提供 `face_ids` NPY 与 `annotations` JSON，跳过对应推理。
逐面 ID 为非负整数；标注含 `object.dimension_cm`、可选 `object.mass_g` 与全部部件的材料、
摩擦和 `mask_color`。`source_up_axis` 明确为 Y 或 Z，最终包采用米制 Z-up。

模型源码只在所属 worker 中导入，各解释器安装同一包。标注接口由使用者配置；
远程调用需要显式 `allow_remote_models`，凭证从指定环境变量读取。

## 合成 mesh 复现

在基础环境运行以下命令，重复复现请更换目录名：

```bash
python -m rfw_w.assets.examples.synthetic_mesh --out outputs/synthetic-input-001
rfw-w asset generate --request outputs/synthetic-input-001/request.json --config outputs/synthetic-input-001/runtime.json --out outputs/synthetic-asset-001
rfw-w asset validate --bundle outputs/synthetic-asset-001/package --out outputs/synthetic-check-001 --backend mujoco --steps 240
```

示例提供 10 cm 立方体、逐面标签与演示物理标注，实际运行 CoACD、物理计算和三格式装配。
默认使用当前解释器；独立物理环境用 `--python /path/to/geometry-env/bin/python`。
该合成输入不代表真实语义或抓取验收。

## 目录与检索

```bash
python -m rfw_w.assets.commands.catalog --release-root /path/to/release --out outputs/catalog-001.db --taxonomy /path/to/taxonomy.json
SE_INDEX_DIR=/path/to/index SE_EXPORT_ROOT=/path/to/release python -m rfw_w.assets.catalog.server.app
```

构建器只读 v1.3 发布目录，校验 canonical leaf 与物理/标注；重分类通过身份映射关联。
bge-m3 从本地加载，目录索引与权重须匹配。资产与场景检索尚未统一后端和索引契约。

## 验证范围

```bash
python -m pytest tests/assets
```

实际 prompt 模型链与四后端资产加载、物理步进已运行，详见 [复现范围](../public-reproduction-spec.md)。
`package_complete` / `_SUCCESS` 表示装配完成，仿真证据独立记录。
正式生成链进行几何检查；独立视觉质量判定仍为 `visual_quality: not_run`。
物理属性是模型估计，完整五后端矩阵、settle、渲染和机器人任务需各自验收。
三格式保留纹理与相对引用，测试覆盖 GLB 往返、目录迁移、MuJoCo 和 USD 纹理关联。
