# ADR-008：保留 pinned 窗户的透明/反射权重

## Context

SPEC-007 双图对照发现 baked 窗户变成黑色开口；不能将 PBR 通道齐全当作外观通过。

## Evidence

Infinigen commit `892947be71c4ff102c6a558d088ffe149561c11f`：
`infinigen/assets/objects/windows/window.py::shader_window_glass` 使用 Transparent、Glossy
和常数 Mix；仅相机光线含反射，其他光线完全透过。
`infinigen/tools/export.py::process_glass_materials` 按名字含 glass 重建为 transmission=1，
丢掉原 Mix 权重。实测节点证据：`outputs/_probe/spec007_normal_ablation_20260922_01/nodes.log`，
原始详细节点也在 export 05 的 material_manifest.json。

## Options

1. 继续全透明：已确认丢失可见反射。
2. 引入 renderer 专用材质：保真空间更大，但增加交付依赖。
3. 对已识别的 pinned 节点图，使用 PreviewSurface 可表达的透明反射近似。

## Decision

采用 3：以 Principled metallic=1、源 Glossy 色彩/roughness、alpha=源反射权重
表达相机看到的透明反射混合。按节点及连接验证，不仅按名称判断；不认识的窗户图具名失败。
保留 mapped 源图、转换参数及近似声明。其他真实玻璃继续单独记录 transmission 近似。

## Rationale

源窗户本来就不折射；保留实际反射混合比，比将所有窗户变成完全透明更接近源意图。
该表示在 Blender 与 USD 都可表达，仍需实际渲染验收。

## Consequences

无法保留仅相机光线反射的分支；阴影/间接光可能不同，金属 Fresnel 也非逐像素等价。
本决定授权实现和测量，不构成视觉验收。SPEC-008 仍负责产品灯光。
