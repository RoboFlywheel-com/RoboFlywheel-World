# SPDX-License-Identifier: Apache-2.0
"""Build the public Pages tree from the catalog, never by copying the review directory."""

import argparse
import json
import re
import shutil
from pathlib import Path

from serve import SITE


def build(media, out):
    text = (media / "catalog.js").read_text()
    prefix = "window.RFW_CATALOG = "
    if not text.startswith(prefix):
        raise ValueError("Unexpected catalog")
    catalog = json.loads(text[len(prefix):].strip().removesuffix(";"))
    files = {"viewer.js", "viewer-license.txt", "catalog.js", "film.mp4"}
    def include(token, suffixes):
        if not re.fullmatch("[0-9a-f]{32}", token):
            raise ValueError("Expected an anonymous media key")
        files.update(token + ext for ext in suffixes)
    for asset in catalog["assets"]:
        for mode in ("visual", "collision"):
            include(asset[mode], (".glb", ".webp"))
    for scene in catalog["scenes"]:
        include(scene["id"], (".mp4", ".webp"))
    out.mkdir(parents=True, exist_ok=False)
    (out / "media").mkdir()
    for name in sorted(files):
        shutil.copyfile(media / name, out / "media" / name)
    for name in ("index.html", "page.js", "style.css", "assets/project-demo.webp",
                 "assets/rigid-construction.webp", "assets/task-scene-construction.webp"):
        (out / name).parent.mkdir(exist_ok=True)
        shutil.copyfile(SITE / name, out / name)
    root = SITE.parents[1]
    for name in ("LICENSE", "NOTICE"):
        shutil.copyfile(root / name, out / name)
    (out / ".nojekyll").touch()
    print(f"{len(catalog['assets'])} assets, {len(catalog['scenes'])} scenes, {len(files)} media files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--media", type=Path, default=SITE / "preview-media.local")
    args = parser.parse_args()
    build(args.media, args.output)
