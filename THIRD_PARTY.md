# Third-party notices

RFW-W is licensed under Apache-2.0. That license does not replace the terms of
third-party software, model weights, datasets, or asset libraries. Components
installed by a package manager can also have transitive dependencies; review
the licenses in each runtime environment before redistribution.

## Adapted source distributed with RFW-W

The following files adapt code from the EmbodiedGen project's Infinigen runtime
and geometry implementations:

- `src/rfw_w/assets/geometry/coacd.py`
- `src/rfw_w/scenes/workers/blender/infinigen_runtime.py`

Copyright (c) 2025 Horizon Robotics. Licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Their source
copyright and license notices are retained. This attribution concerns the
adapted EmbodiedGen code, not the Infinigen project itself.

## External software

These components are obtained separately rather than relicensed as part of
RFW-W. Follow the linked upstream project and retain its notices when
redistributing the component.

| Component | RFW-W use | Upstream license / terms |
| --- | --- | --- |
| [three.js r169](https://github.com/mrdoob/three.js/tree/r169) | Legacy catalog-viewer build artifacts only; this source release contains no catalog viewer or vendored JavaScript | MIT; Copyright 2010-2024 Three.js Authors |
| [Infinigen](https://github.com/princeton-vl/infinigen) | Optional, independently configured room generator | BSD-3-Clause; distinct from the Apache-2.0 EmbodiedGen adaptations above |
| [OpenUSD / usd-core](https://github.com/PixarAnimationStudios/OpenUSD) | Optional USD import and export | Apache-2.0 |
| [trimesh](https://github.com/mikedh/trimesh) | Mesh processing and exchange | MIT |
| [CoACD](https://github.com/SarahWeiii/CoACD) | Convex decomposition | MIT |
| [PyTorch](https://github.com/pytorch/pytorch) | Model inference and retrieval | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers) | Model inference and retrieval | Apache-2.0 |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | Optional annotation service client | Apache-2.0 |
| [DashScope Python SDK](https://github.com/aliyun/dashscope-sdk-python) | Optional compatible model-service integration; not a direct package dependency | Apache-2.0 |

No three.js files are redistributed in this revision. If the legacy viewer is
reintroduced, the MIT notice must accompany `three.module.js` and the
`GLTFLoader.js`, `OrbitControls.js`, and `BufferGeometryUtils.js` addons.

## Model weights

Model weights are external inputs and are not included in this repository.
Record the exact revision, file hashes, and applicable terms for each download.
A source-code license does not automatically cover that project's weights.

| Model or project | License guidance |
| --- | --- |
| [FLUX.2-klein-4B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B) | The documented 4B release uses Apache-2.0; verify the exact downloaded revision |
| [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | Use is governed by the upstream model card and its usage restrictions; check commercial-use eligibility separately |
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Source code is MIT; review the separate terms of every downloaded weight and preprocessing model |
| [P3-SAM / Hunyuan3D-Part](https://huggingface.co/tencent/Hunyuan3D-Part) | Review the upstream code and weight terms before use or redistribution |
| [Sonata](https://github.com/facebookresearch/sonata) | Review the upstream code and checkpoint terms before use or redistribution |
| [DINOv3](https://github.com/facebookresearch/dinov3) | Review the upstream code and weight terms before use or redistribution |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | Apache-2.0; verify the pinned revision |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3) | MIT; verify the pinned revision |

## Datasets and asset libraries

RFW-W source code does not grant redistribution rights for any dataset or
asset library. Small synthetic fixtures in this repository are not samples of
the external libraries.

- **HSSD:** an optional external furniture source. Review and follow the
  [upstream HSSD terms](https://github.com/facebookresearch/hssd) before use.
- **SimReady / RFW v1.3 and later releases:** distributed separately, if at
  all. Each release must carry its own provenance, asset-level license records,
  inventory, and redistribution terms. The runtime value `asset-license:
  unrecorded` means that permission is not established; it is not a license.
- **RFW-R / RFW-A / RFW-S datasets:** future or separate data releases require
  their own license notices. The Apache-2.0 source license and statements about
  forthcoming releases do not license the data.
- **Poly Haven and ambientCG:** optional public material sources; their public
  asset terms are CC0. Retain source provenance and verify the terms of each
  downloaded item.

Rendering, loading, or simulation validation of an external asset is technical
evidence only and does not establish permission to publish that asset.
