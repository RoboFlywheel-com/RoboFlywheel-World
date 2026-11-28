# SPDX-License-Identifier: Apache-2.0
"""Canonical release leaf lookup; object IDs require explicit mapping."""

from pathlib import Path
import re


def resolve(root, key):
    if not isinstance(key, str) or not re.fullmatch(r"[0-9]+_[0-9]+_[0-9]{4,}", key):
        raise ValueError("invalid release leaf")
    domain, category, _ = key.split("_")
    root = Path(root).resolve()
    path = (root / "assets" / domain / category / key).resolve()
    if not path.is_relative_to(root) or not (path / "_SUCCESS").is_file():
        raise FileNotFoundError("complete release leaf is unavailable")
    return path


def visual_mesh(asset_dir, key):
    """Read the mesh actually referenced by the asset's visual URDF."""
    import xml.etree.ElementTree as ET

    directory = Path(asset_dir).resolve()
    urdfs = sorted(directory.glob("*.urdf"))
    for urdf in urdfs:
        item = ET.parse(urdf).find(".//visual/geometry/mesh")
        if item is not None:
            name = item.get("filename")
            target = (urdf.parent / name).resolve()
            if not target.is_relative_to(directory) or not target.is_file():
                raise ValueError("visual mesh reference escapes asset package")
            return target
    raise FileNotFoundError("asset lacks an explicit visual mesh reference")


def collision_meshes(asset_dir, key):
    import xml.etree.ElementTree as ET

    directory = Path(asset_dir).resolve()
    result = []
    for urdf in sorted(directory.glob("*.urdf")):
        for item in ET.parse(urdf).findall(".//collision/geometry/mesh"):
            target = (urdf.parent / item.get("filename")).resolve()
            if not target.is_relative_to(directory) or not target.is_file():
                raise ValueError("collision reference escapes asset package")
            result.append(target)
        if result:
            return result
    raise FileNotFoundError("asset lacks explicit collision mesh references")
