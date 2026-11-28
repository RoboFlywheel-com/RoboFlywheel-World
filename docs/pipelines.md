# RFW-W 资产与场景管线

RFW-W 由一个安装包内的资产模块、场景模块与公共命令组成，数据和权重在仓库外。
完整布局见 [模块设计](module-design.md)，安装步骤见 [环境与复现](reproduction.md)，请求格式见两侧 README。

## 安装

```bash
python -m pip install -e .
rfw-w doctor
rfw-w asset generate --help
rfw-w scene generate --help
```

核心安装只提供运行与契约。科学计算、模型、浏览和仿真依赖在根级 extras 中按功能安装，
不同模型/引擎可使用不同环境。公共命令用 `RFW_ASSET_PYTHON`、`RFW_SCENE_PYTHON`
或 `rfw-w --python /path/to/python …` 选择 worker 解释器；每个环境安装同一个 RFW-W 包。

## 两条管线

| 用例 | 输入与流程 | 产物 |
| --- | --- | --- |
| 单资产生成 | prompt/image/mesh → 几何检查 → 分割/标注 → 碰撞/物理 → 装配 | 独立资产包与逐阶段记录 |
| 场景生成 | SceneSpec/任务 → 检索/准备 → 核心/房间/外围 → 修复/交付 | SceneIR、delivery 与证据 |

示例预览不触发模型、资产检索或仿真：

```bash
rfw-w scene generate --experiment office_pick_cup --compile-only --output-dir outputs/preview-001
```

这是任务/房间预览，不是完整场景或仿真验收。实际模型调用须明确许可与本机资源配置。

新生成的独立资产包可用 `--asset-bindings` 与显式 `--asset-license` 绑定为任务目标；
两实体规格使用构造支撑体，直接绕过发布全库及语义检索。
支撑体/上下文的直接绑定与原生多hull场景物理仍是缺口，详见场景 README。

## v1.3 发布资产

```text
assets/<domain>/<class>/<leaf>/
  m_<leaf>.urdf
  coacd_multi.urdf
  <leaf>.usd
  part_physics.json
  mesh/
  mjcf/
  affordance/
  _SUCCESS
```

leaf 为 `<domain>_<class>_<sequence>`，sequence 至少四位、左侧补零，超过 9999 保留完整编号。
生产 object_id 与重分类 leaf 用 release_mapping 关联，
不能拼接旧 key。物理来自 sidecar，摆放使用实测尺寸，坐标准备保持记录。
`_SUCCESS` 不等于五后端验证成功。

资产 catalog 保留 FTS5/bge-m3/RRF 检索；场景原生目录/SBERT 检索仍是另一种实现，
两者尚未完成共用 schema 与后端收敛，配置前核对索引版本、模型身份和 key mapping。

## 检查

```bash
bash scripts/check.sh
bash scripts/check.sh --stdlib-only
bash scripts/dev/test-scenes.sh --assets --require-assets
bash scripts/dev/test-scenes.sh --delivery
python scripts/scan_sensitive.py
```

独立检索：`rfw-w scene retrieve --help`。本地索引构建：
`python -m rfw_w.scenes.preparation.build_simready_index --help`、
`python -m rfw_w.scenes.preparation.build_semantic_index --help`。

真实发布库对照使用 `RFW_RELEASE_ROOT`，独立生成包运行 `rfw-w asset validate --help`；v1.3 发布 leaf 运行
`python scripts/audit_release_asset.py --key <leaf> --out <new-dir> --mujoco-python <python>`。
输出新目录，外部全库保持只读。缺资源、未执行仿真分别报告，不继承历史验收。
