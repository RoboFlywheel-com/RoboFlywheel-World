# SPDX-License-Identifier: Apache-2.0
"""Batch schedules the same independently recorded single-asset use case."""

from pathlib import Path
from .application import generate
from .contracts.request import AssetRequest


def generate_batch(requests, runtime, out):
    requests = [
        AssetRequest.from_dict(r) if isinstance(r, dict) else r for r in requests
    ]
    if len({r.asset_id for r in requests}) != len(requests):
        raise ValueError("duplicate batch asset_id")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for request in requests:
        try:
            results.append(generate(request, runtime, out / request.asset_id))
        except Exception as exc:
            results.append(
                {
                    "asset_id": request.asset_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
            )
    return results
