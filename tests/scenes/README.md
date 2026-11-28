# RFW-W 场景测试

从仓库根运行，普通 Python 清除 Isaac 的 `PYTHONPATH`、`PYTHONHOME` 注入。

```bash
bash scripts/dev/test-scenes.sh
bash scripts/dev/test-scenes.sh --stdlib-only
bash scripts/dev/test-scenes.sh --assets --require-assets
bash scripts/dev/test-scenes.sh --delivery
```

| 检查 | 输入要求 |
| --- | --- |
| unit / contract | 默认源码检查；模型和 Isaac 用例不默认执行 |
| stdlib-only | 已安装 RFW-W 的独立标准库解释器 |
| assets | 真实资产与索引；`--require-assets` 将缺资源升级为失败 |
| delivery | 已生成的当前交付；无产物时失败 |
| Isaac / VLM | 显式运行 `test-isaac.sh` / `test-vlm.sh`，配置对应环境与资源 |

解释器由 `PVSCENE_RUNTIME_PYTHON`、`PVSCENE_STDLIB_PYTHON` 配置。
分别报告 passed、skipped、not_run、blocked；声明一致性不等于物理或任务成功。
