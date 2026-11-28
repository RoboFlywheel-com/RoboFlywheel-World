# RFW-W 单资产示例

`request.json` 提供 prompt 起点。复制 `runtime.example.json` 到本机配置，填写阶段解释器、
模型资源和服务接口；安装方法见 [复现指南](../../docs/reproduction.md)。
标注服务通过 `endpoint`、`model`、`api-key-env` 配置，调用开关为 `allow_remote_models`。

```bash
rfw-w asset generate --request examples/assets/request.json --config /path/to/runtime.json --out outputs/asset-001
```

已有 mesh 可提供 `face_ids` 与 `annotations`，仅运行几何、物理和导出阶段。
每次运行使用新的输出目录。
