# SPDX-License-Identifier: Apache-2.0
"""RFW-W asset batch entry."""

import sys
from rfw_w.assets.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["batch", *sys.argv[1:]]))
