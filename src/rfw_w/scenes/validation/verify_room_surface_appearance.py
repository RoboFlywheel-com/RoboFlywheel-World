# SPDX-License-Identifier: Apache-2.0
"""Independent process: verify a saved appearance bundle without library access."""
import argparse
import json
from pathlib import Path
import sys
from rfw_w.scenes.geometry.surface_appearance import verify_surface_appearance
from rfw_w.scenes.contracts.surface_appearance import SurfaceAppearanceError


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    try: report=verify_surface_appearance(a.delivery)
    except (SurfaceAppearanceError,OSError,ValueError,KeyError,TypeError) as e:
        report={'verified':False,'problems':[{'code':getattr(e,'code','SURFACE_BINDING_INVALID'),'error_type':type(e).__name__}]}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    return 0 if report['verified'] else 1
if __name__=='__main__':raise SystemExit(main())
