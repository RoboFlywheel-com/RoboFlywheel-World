# SPDX-License-Identifier: Apache-2.0
"""Independent RFW-W asset package verification."""

import argparse, json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args()
    from rfw_w.assets.simulation.validation import validate

    print(json.dumps(validate(args.bundle, args.out, args.backend, args.steps)))


if __name__ == "__main__":
    main()
