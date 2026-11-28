# SPDX-License-Identifier: Apache-2.0
"""Building the montage the Visual Critic is shown.

Architecture §9 specifies a montage rather than a pile of full-resolution frames, and the reason is
practical as well as token-economic: five 1280x720 PNGs is tens of megabytes, and sending them one
by one timed out a real run mid-request and took the whole acceptance with it. Tiling downscaled
views into one labelled image keeps the critic's attention on composition -- what it is asked to
judge -- and keeps the request small enough to complete.

Labels are burned into the image because the critique references views by name, and a critic that
cannot see which panel is which cannot say where a defect is.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Sequence

TILE_WIDTH = 640
GAP_PX = 8
LABEL_HEIGHT_PX = 26
BACKGROUND = (24, 26, 30)
LABEL_COLOUR = (235, 235, 235)


class MontageError(RuntimeError):
    """The montage could not be built."""


def build_montage(
    views: Sequence[tuple[str, Path]],
    *,
    columns: int = 3,
    tile_width: int = TILE_WIDTH,
) -> tuple[bytes, list[str]]:
    """Tile labelled, downscaled views into one PNG. Returns ``(png_bytes, labels_used)``.

    Views are scaled to a common width and arranged in rows; a view that cannot be read is skipped
    with its label returned separately rather than silently dropped, so the caller can say which
    view the critic never saw.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - PIL is present in the runtime
        raise MontageError(f"building a montage needs Pillow: {error}") from error

    if not views:
        raise MontageError("a montage needs at least one view")
    columns = max(1, min(columns, len(views)))

    tiles = []
    skipped: list[str] = []
    for label, path in views:
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, ValueError):
            skipped.append(label)
            continue
        if image.width != tile_width:
            height = max(1, round(image.height * tile_width / image.width))
            image = image.resize((tile_width, height), Image.LANCZOS)
        tiles.append((label, image))
    if not tiles:
        raise MontageError("no view could be read for the montage")

    rows = (len(tiles) + columns - 1) // columns
    tile_height = max(image.height for _, image in tiles) + LABEL_HEIGHT_PX
    canvas = Image.new(
        "RGB",
        (columns * tile_width + (columns + 1) * GAP_PX,
         rows * tile_height + (rows + 1) * GAP_PX),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(tiles):
        column, row = index % columns, index // columns
        x = GAP_PX + column * (tile_width + GAP_PX)
        y = GAP_PX + row * (tile_height + GAP_PX)
        draw.text((x + 2, y + 5), label, fill=LABEL_COLOUR)
        canvas.paste(image, (x, y + LABEL_HEIGHT_PX))

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), [label for label, _ in tiles]


def montage_data_url(views: Sequence[tuple[str, Path]], *, columns: int = 3,
                     tile_width: int = TILE_WIDTH) -> tuple[str, list[str]]:
    """The montage as a data URL, ready for a chat-completions image part."""
    payload, labels = build_montage(views, columns=columns, tile_width=tile_width)
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii"), labels
