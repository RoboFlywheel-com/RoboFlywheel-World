# SPDX-License-Identifier: Apache-2.0
"""Validated single-asset production requests; paths are local inputs only."""

from __future__ import annotations
from dataclasses import dataclass
import re
from pathlib import Path

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def identifier(value):
    if (
        not isinstance(value, str)
        or not _IDENTIFIER.fullmatch(value)
        or value in (".", "..")
    ):
        raise ValueError("asset_id and class_id must be safe identifiers")
    return value


@dataclass(frozen=True)
class AssetRequest:
    asset_id: str
    class_id: str
    input_kind: str
    input_value: str
    category: str = ""
    seed: int = 2025
    annotations: str | None = None
    face_ids: str | None = None
    formats: tuple[str, ...] = ("urdf", "mjcf", "usd")
    mass_policy: str = "anchor_on_flag"
    source_up_axis: str = "Y"

    def __post_init__(self):
        identifier(self.asset_id)
        identifier(self.class_id)
        if self.input_kind not in ("prompt", "image", "mesh"):
            raise ValueError("input must specify exactly one of prompt, image, mesh")
        if not isinstance(self.input_value, str) or not self.input_value.strip():
            raise ValueError("input must not be empty")
        if not isinstance(self.category, str):
            raise ValueError("category must be a string")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        if (
            not isinstance(self.formats, (tuple, list))
            or not self.formats
            or any(
                not isinstance(item, str) or item not in ("urdf", "mjcf", "usd")
                for item in self.formats
            )
            or len(set(self.formats)) != len(self.formats)
        ):
            raise ValueError("unsupported or duplicate export format")
        object.__setattr__(self, "formats", tuple(self.formats))
        if self.mass_policy not in ("composed", "anchor_vlm", "anchor_on_flag"):
            raise ValueError("unsupported mass policy")
        if self.source_up_axis not in ("Y", "Z"):
            raise ValueError("source_up_axis must be Y or Z")
        local_inputs = ("annotations", "face_ids") + (
            ("input_value",) if self.input_kind != "prompt" else ()
        )
        for name in local_inputs:
            value = getattr(self, name)
            if value is None and name != "input_value":
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(name + " must be a file path")
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(name + " file does not exist")
            object.__setattr__(self, name, str(path.resolve()))

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("request must be an object")
        required = ("asset_id", "class_id", "input")
        allowed = set(required) | {
            "category",
            "seed",
            "annotations",
            "face_ids",
            "formats",
            "mass_policy",
            "source_up_axis",
        }
        if set(data) - allowed:
            raise ValueError("unknown asset request fields")
        if any(k not in data for k in required):
            raise ValueError("request requires asset_id, class_id, input")
        inputs = data["input"]
        if not isinstance(inputs, dict) or len(inputs) != 1:
            raise ValueError("input must specify exactly one of prompt, image, mesh")
        kind, value = next(iter(inputs.items()))
        return cls(
            data["asset_id"],
            data["class_id"],
            kind,
            value,
            data.get("category", data["class_id"]),
            data.get("seed", 2025),
            data.get("annotations"),
            data.get("face_ids"),
            data.get("formats", ("urdf", "mjcf", "usd")),
            data.get("mass_policy", "anchor_on_flag"),
            data.get("source_up_axis", "Y"),
        )
