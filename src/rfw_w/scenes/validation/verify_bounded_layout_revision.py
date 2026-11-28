# SPDX-License-Identifier: Apache-2.0
"""Independent saved-source, scope, actual geometry and resource replay."""
import argparse
import json
from pathlib import Path
from rfw_w.scenes.orchestrator.layout_revision_tools import verify_layout_revision,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--delivery',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    try:report=verify_layout_revision(a.delivery)
    except (ValueError,OSError,KeyError,TypeError) as error:
        report={'verified':False,'failure_code':getattr(error,'code','LAYOUT_INPUT_INVALID'),'detail':str(error)}
    write(a.output_dir,'report.json',report);print(json.dumps(report));return 0 if report['verified'] else 1


if __name__=='__main__':raise SystemExit(main())
