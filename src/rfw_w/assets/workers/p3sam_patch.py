# SPDX-License-Identifier: Apache-2.0
"""P3-SAM construction patch for cached Sonata weights and no-JIT environments."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def patch_p3sam(p3sam_root: str, *, sonata_ckpt: str | None = None) -> None:
    if sonata_ckpt is not None and not Path(sonata_ckpt).is_file():
        raise FileNotFoundError("Local Sonata checkpoint unavailable")
    import torch
    import torch.nn as nn

    root = os.path.abspath(p3sam_root)
    hunyuan_part_root = os.path.dirname(root)
    for path in (root, os.path.join(hunyuan_part_root, "XPart", "partgen")):
        if path not in sys.path:
            sys.path.insert(0, path)

    from models import sonata

    p3sam_model = importlib.import_module("model")

    def build_p3sam(self):
        self.sonata = sonata.load(
            sonata_ckpt or "sonata",
            repo_id="facebook/sonata",
            download_root=os.path.expanduser("~/.cache/huggingface/hub/sonata"),
        )
        self.mlp = nn.Sequential(
            nn.Linear(1232, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )
        self.transform = sonata.transform.default()
        for index in range(1, 4):
            setattr(
                self,
                f"seg_mlp_{index}",
                nn.Sequential(
                    nn.Linear(512 + 3 + 3, 512),
                    nn.GELU(),
                    nn.Linear(512, 512),
                    nn.GELU(),
                    nn.Linear(512, 1),
                ),
            )
        self.seg_s2_mlp_g = nn.Sequential(
            nn.Linear(512 + 3 + 3 + 3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        for index in range(1, 4):
            setattr(
                self,
                f"seg_s2_mlp_{index}",
                nn.Sequential(
                    nn.Linear(512 + 3 + 3 + 3 + 256, 256),
                    nn.GELU(),
                    nn.Linear(256, 256),
                    nn.GELU(),
                    nn.Linear(256, 1),
                ),
            )
        self.iou_mlp = nn.Sequential(
            nn.Linear(512 + 3 + 3 + 3 + 256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.iou_mlp_out = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )
        self.iou_criterion = torch.nn.MSELoss()

    p3sam_model.build_P3SAM = build_p3sam
