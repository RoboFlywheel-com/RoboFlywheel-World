# Local project page

Run `python scripts/project_page/serve.py --port 8081` from the repository root and open loopback port 8081. The interactive atlas requires HTTP and WebGL. There is no external viewer service: the Three.js-based model-viewer runtime and display models load from the preview server. The [visual checklist](visual-checklist.md) records remaining evidence and publication requirements.

The HTML, CSS and JS are portable. The selected review images and videos live **outside Git**. On a fresh checkout without a review bundle, the overview figure and documentation remain available but the atlas and players are not populated.

## Series template

The layout and stylesheet adapt the [RFW-Soft project page source](https://github.com/roboflywheel-com/RoboFlywheel-Soft/tree/29be8b322db78565c4ddb82a69c29ed4bd284cdb) from its published `gh-pages` branch. Copyright 2026 RFW-S contributors, Apache-2.0. RFW-W provides its own media, catalog, scene player, comparison controls and resource links. The reference project’s research results and author list are not part of this page.

## Rebuild review media

Each candidate batch retains a private review report. The exporter preserves UV vertices, repairs winding and regenerates normals before reloading both GLBs for six-view depth checks. Review the candidate thumbnails for recognizable shape and category agreement. Run `scripts/project_page/finalize.py` on the reviewed batches with `--exclude` (a private JSON map from display name to rejection reason) and a fresh `--output`. Point the local preview link at that final output, not at an unreviewed batch.

For a curated opening grid, pass `--featured` with an ordered JSON list of objects containing the review `name` and the public `title`. Its first entry becomes the initial interactive model. Use appearance posters to balance shapes, colors and patterns; the collision palette is illustrative. Other admitted items follow in family rotation.

Use a separate Python environment with NumPy, trimesh, Pillow and pyrender (with a working EGL renderer); the page itself requires none of these. Configure read-only access using the local OSS CLI. Set `RFW_PAGE_RELEASE_URI` to the v1.3 release root, `PAGE_FILM` to the full project demo, `PAGE_SCENE_CATALOG` to the scene clip catalog, and `PAGE_OUTPUT` to a **new external directory**. When assembling older batches, pass the original full demo to the finalizer with `--film`. From the repository root:

```bash
env -u PYTHONHOME -u PYTHONPATH python scripts/project_page/prepare.py \
  --film "$PAGE_FILM" --scene-catalog "$PAGE_SCENE_CATALOG" --output "$PAGE_OUTPUT" \
  --selection "$PRIVATE_SELECTION" --viewer-package "$VIEWER_PACKAGE"
```

The selection is an external private CSV with columns `production,name,family`; only the last two fields are public display labels. Supply the extracted npm package for `@google/model-viewer@4.1.0` (including its license). The preparer emits standalone randomly named GLBs and posters; source identities, mesh names, external texture URIs and source packages are not served. Preview geometry can still be retrieved by anyone granted browser access. Sizes are normalized per asset; hull colors separate components rather than materials. Scene recordings are not proof that the displayed assets were bound in those scenes.
