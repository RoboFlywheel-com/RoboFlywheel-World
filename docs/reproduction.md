# RFW-W 环境安装与复现

两条管线共用一个包，几何、模型与仿真 worker 分环境安装。以下命令从仓库根运行；
`/path/to/...` 为使用者指定的位置。普通 Python 清除 `PYTHONPATH` 与 `PYTHONHOME`
中的 Isaac 注入，各环境安装同一份 RFW-W wheel。

## 基础环境

已测 Linux x86_64、Python 3.10。版本约束见
[geometry.constraints.txt](../examples/reproduction/geometry.constraints.txt)。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -c examples/reproduction/geometry.constraints.txt \
  '.[asset-geometry,scene-geometry,usd,mujoco]' pybullet sapien
python -m pip check
python -m pip wheel --no-deps --wheel-dir dist .
rfw-w doctor
```

单资产、场景几何、CoACD 与格式导出使用此环境。测试依赖单独安装 `.[test]`。
模型和引擎环境安装 `dist/` 中生成的 wheel，不使用全局 `PYTHONPATH` 拼接外部项目。

## 模型环境

| 阶段 | Python / 已测版本 | 资源 |
| --- | --- | --- |
| image | 3.11；Torch 2.7.0/cu128、diffusers 0.40.0、transformers 5.16.0 | FLUX.2-klein-4B |
| background | 3.11；Torch 2.7.0/cu128、transformers 4.57.6 | RMBG-2.0、torchvision、timm |
| 3d | 3.10；Torch 2.7.0/cu128、transformers 4.57.6、xformers 0.0.30 | TRELLIS.2、DINOv3、RMBG 与 CUDA 扩展 |
| segmentation | 3.10；Torch 2.5.1/cu118、spconv 2.3.8、cumm 0.7.11 | P3-SAM、Sonata |
| render | 基础环境；Blender 4.2 | 本地 Blender 可执行文件 |
| annotation | 3.10；openai、json-repair、NumPy、trimesh、Pillow | 使用者配置的多模态服务 |

### 图像生成

版本见 [flux.constraints.txt](../examples/reproduction/flux.constraints.txt)。
图像环境独立于 TRELLIS 和去背环境，避免不同 transformers 与 attention 扩展混用。

```bash
python3.11 -m venv /path/to/rfw-image
/path/to/rfw-image/bin/python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
/path/to/rfw-image/bin/python -m pip install -c examples/reproduction/flux.constraints.txt \
  diffusers transformers huggingface-hub accelerate modelscope
/path/to/rfw-image/bin/python -m pip install --no-deps dist/rfw_w-0.1.0.dev0-py3-none-any.whl
/path/to/rfw-image/bin/python -m pip check
```

从 ModelScope 获取官方 [FLUX.2-klein-4B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B)
Diffusers 目录，填入 `stages.image.options.ckpt`：

```python
from modelscope import snapshot_download

snapshot_download(
    "black-forest-labs/FLUX.2-klein-4B",
    revision="master",
    local_dir="/path/to/FLUX.2-klein-4B",
    ignore_file_pattern=["flux-2-klein-4b.safetensors"],
)
```

根级单文件权重与 Diffusers 目录重复，可跳过。记录实际下载 revision 与文件哈希；
可变 `master` 不能代替固定版本。4B 模型使用 Apache-2.0。

### 去背与部件标注

```bash
python3.11 -m venv /path/to/rfw-background
/path/to/rfw-background/bin/python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
/path/to/rfw-background/bin/python -m pip install transformers==4.57.6 timm numpy pillow
/path/to/rfw-background/bin/python -m pip install --no-deps dist/rfw_w-0.1.0.dev0-py3-none-any.whl
/path/to/rfw-background/bin/python -m pip check

python3.10 -m venv /path/to/rfw-annotation
/path/to/rfw-annotation/bin/python -m pip install '.[asset-annotation]'
/path/to/rfw-annotation/bin/python -m pip check
```

下载 RMBG-2.0 到外部模型目录。标注服务由使用者配置，runtime 只提供
`endpoint`、`model`、`api-key-env` 及 `allow_remote_models` 入口，见
[runtime.example.json](../examples/assets/runtime.example.json)。接口需要图像输入支持；
物理属性标注是模型估计，后续物理计算与验证独立执行。

### 三维生成

获取 [TRELLIS.2 源码](https://github.com/microsoft/TRELLIS.2)，本轮使用 commit
`75fbf0183001ed9876c8dbb35de6b68552ee08bd`。源码、权重和扩展存放在仓库外。

```bash
python3.10 -m venv /path/to/rfw-3d
/path/to/rfw-3d/bin/python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
/path/to/rfw-3d/bin/python -m pip install xformers==0.0.30 --no-deps \
  --index-url https://download.pytorch.org/whl/cu128
/path/to/rfw-3d/bin/python -m pip install 'numpy<2' scipy pillow transformers==4.57.6 \
  timm kornia easydict imageio imageio-ffmpeg opencv-python-headless tensorboard pandas \
  zstandard lpips ninja wheel 'setuptools>=64' plyfile moderngl
/path/to/rfw-3d/bin/python -m pip install --no-deps dist/rfw_w-0.1.0.dev0-py3-none-any.whl
```

源码 checkout 与扩展安装：

```bash
git clone https://github.com/microsoft/TRELLIS.2.git /path/to/TRELLIS.2
git -C /path/to/TRELLIS.2 checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
/path/to/rfw-3d/bin/python -m pip install \
  git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
source /path/to/rfw-3d/bin/activate
cd /path/to/TRELLIS.2
bash setup.sh --cumesh --flexgemm --o-voxel --nvdiffrast
python -m pip check
```

该命令在外部源码目录执行；后续 RFW-W 命令回到仓库根。
扩展选项来自 [固定版本的 setup.sh](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/setup.sh)。
CUDA 扩展必须针对同一 Torch/ABI 构建；需要 CUDA 12.8 Toolkit、C/C++ 编译器、Python 开发头文件和匹配的 C++ 运行库。
Triton 首次运行也需要可用编译器。

配置 `ATTN_BACKEND=sdpa`、`SPARSE_ATTN_BACKEND=xformers`、`OMP_NUM_THREADS=4`、
`MKL_NUM_THREADS=4`。在独立部署副本中设置 DINOv3/RMBG 的本地路径，保留原始权重。
512 端测使用 `pipeline-type: "512"`、`texture-size: 512`、`decimation: 20000`；
稀疏 attention 不支持 sdpa，历史 TRELLIS 环境不能直接视为 TRELLIS.2 环境。

### 部件分割

获取 [P3-SAM 源码](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/tree/main/P3-SAM)、[模型权重](https://huggingface.co/tencent/Hunyuan3D-Part)与
[Sonata](https://github.com/facebookresearch/sonata) 权重。版本见
[p3sam.constraints.txt](../examples/reproduction/p3sam.constraints.txt)。

```bash
python3.10 -m venv /path/to/rfw-segmentation
/path/to/rfw-segmentation/bin/python -m pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
/path/to/rfw-segmentation/bin/python -m pip install \
  -c examples/reproduction/p3sam.constraints.txt \
  torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu118.html \
  spconv-cu118 cumm-cu118 numpy scipy trimesh scikit-learn fpsample numba \
  timm addict safetensors omegaconf einops
/path/to/rfw-segmentation/bin/python -m pip install --no-deps dist/rfw_w-0.1.0.dev0-py3-none-any.whl
/path/to/rfw-segmentation/bin/python -m pip check
```

`sonata-ckpt` 显式指向本地权重。spconv 2.3.8 使用 cumm 0.7.11；避免混入 0.8 系列。
P3-SAM 标签对应模型清理后的 mesh，导出保持源尺度与坐标。

## 仿真与检索环境

| 后端 | 安装方式 / 已测版本 | 验证范围 |
| --- | --- | --- |
| MuJoCo | 基础环境；3.12.0 | MJCF 加载、原生物理与步进 |
| PyBullet | 基础环境；3.2.7 | URDF 原生惯量与步进 |
| SAPIEN | 基础环境；3.0.3 | CPU 物理，不包含 RGB 验收 |
| Genesis | 独立环境安装 `genesis-world`；已测 1.3.3 | MJCF 物理与 GPU 步进 |
| Isaac Sim | 按对应版本官方安装，再在 worker Python 安装 RFW-W | 逐项运行独立验证 |

Genesis 使用独立环境，先安装与 GPU/驱动匹配的 Torch：

```bash
python3.10 -m venv /path/to/rfw-genesis
# 在该解释器中安装所选 Torch，再安装后端与 RFW-W。
/path/to/rfw-genesis/bin/python -m pip install genesis-world==1.3.3 PyOpenGL==3.1.7
/path/to/rfw-genesis/bin/python -m pip install --no-deps dist/rfw_w-0.1.0.dev0-py3-none-any.whl
/path/to/rfw-genesis/bin/python -m pip check
```

端测使用 Torch 2.13.0/cu130；恢复的历史环境不能作为公开依赖锁。
Isaac 的安装与许可由使用者处理，安装完成不代表已通过验收。

原生语义检索安装 `.[scene-retrieval]`，配置本地 all-mpnet-base-v2 权重和类别索引。
本轮模型 revision 为 `e8c3b32edf5434bc2275fc9bab85f82640a19130`。
配置方式见 [场景指南](scenes/README.md)，目录索引与物理数据使用同一发布版本。

## 运行两条管线

单资产与场景分别使用模板中的本机 runtime。输出目录必须是新目录。

```bash
rfw-w asset generate --request examples/assets/request.json \
  --config /path/to/assets.runtime.json --out outputs/asset-001
rfw-w asset validate --bundle outputs/asset-001/package --out outputs/asset-check-001 --steps 600

rfw-w scene generate --spec /path/to/task.json --asset-bindings /path/to/bindings.json \
  --asset-license unrecorded --output-dir outputs/scene-001 --request-only --no-replan
```

`bindings.json` 将任务目标映射到资产包目录；相对路径以该文件所在目录为基准。
合成 mesh 的完整示例见 [资产指南](assets/README.md)。

组合复现脚本使用 [runtime.example.json](../examples/reproduction/runtime.example.json)：

```bash
python scripts/reproduce.py --config /path/to/reproduction.runtime.json --out outputs/reproduction-001
```

脚本覆盖生成、复制后重验、四后端资产 probe 和两实体场景请求；可配置真实请求与
只读 v1.3 发布对照。`summary.json` 记录每项状态，返回 1 表示部分、失败或未运行。
此脚本不把模型链或缺少 Isaac 的矩阵标为完整成功。

## 已验证范围

- prompt → FLUX → RMBG → TRELLIS.2 → P3-SAM → 渲染 → 标注 → 物理计算 → 三格式资产包。
- 生成包及 v1.3 对照资产的 MuJoCo、PyBullet、SAPIEN、Genesis 加载、mass/COM/inertia 与步进。
- 生成资产绑定两实体场景、独立交付验证与 MuJoCo 600 步请求检查；本地 SBERT 类别检索。
后续全库索引及三种真实场景检查见 [检索检查](scenes/retrieval-evaluation.md)。

场景 probe 使用单 mesh 凸代理、请求质量和引擎估计惯量，仅添加地面。
完整墙体、多 hull、settle、五后端场景和机器人任务需另行验收。
模型参数估计精度与生成质量也需独立评估，运行通过不等于真实任务成功。
