#!/usr/bin/env python3
"""
make_init_sweep.py

Generates the param files and the experiment JSON for a sweep over the initial
guess offsets (filter.init_pose_offset x filter.init_ang_offset).

Every (pos, ang) pair gets its own yaml, copied from the base yaml with only the
two offsets changed (comments and all other values are kept), and its own
experiment in the JSON, named e.g. "pos0.15_ang5". The experiment column of the
results CSV then gives the heatmap cell of every run.

    python3 tools/make_init_sweep.py --bags fscale3_circle [--name init_sweep] [--base params/uwb_imu_ekf.yaml]

writes
    params/<name>/pos<p>_ang<a>.yaml   one per pair
    params/<name>.json                 run with: run_experiments params/<name>.json
"""

import argparse
import json
import re
from pathlib import Path

POS_OFFSETS = [0.0, 0.15, 0.30, 0.5, 1.0]  # [m]
ANG_OFFSETS = [0.0, 5.0, 15.0, 30.0, 45.0]  # [deg]
SEEDS = [1, 2, 3, 4, 5]


def set_value(text, key, value):
    # Replace the value of "key:" and keep the indentation and the trailing comment.
    pattern = re.compile(r"^(\s*" + re.escape(key) + r":\s*)[^\s#]+", re.MULTILINE)
    text, n = pattern.subn(lambda m: m.group(1) + repr(float(value)), text)
    if n != 1:
        raise SystemExit(f"expected exactly one '{key}:' in the base yaml, found {n}")
    return text


def label(pos, ang):
    return f"pos{pos:.2f}_ang{ang:g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bags", nargs="+", required=True, help="data folders in test_data/csv to run every pair on")
    parser.add_argument("--name", default="init_sweep", help="name of the param subfolder and the JSON")
    parser.add_argument("--base", default="params/uwb_imu_ekf.yaml", help="yaml the sweep files are copied from")
    parser.add_argument("--params-dir", default="params")
    parser.add_argument("--save-plots", action="store_true", help="save the plots of every run")
    args = parser.parse_args()

    params_dir = Path(args.params_dir)
    yaml_dir = params_dir / args.name
    yaml_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base).read_text()

    experiments = {}
    for pos in POS_OFFSETS:
        for ang in ANG_OFFSETS:
            name = label(pos, ang)
            text = set_value(base, "init_pose_offset", pos)
            text = set_value(text, "init_ang_offset", ang)
            (yaml_dir / f"{name}.yaml").write_text(text)

            # param_file is relative to the params dir
            param_file = f"{args.name}/{name}.yaml"
            experiments[name] = {
                bag: {"param_file": param_file, "runs": len(SEEDS), "run_seeds": SEEDS, "save_plots": args.save_plots}
                for bag in args.bags
            }

    json_file = params_dir / f"{args.name}.json"
    json_file.write_text(json.dumps(experiments, indent=4) + "\n")

    n_runs = len(experiments) * len(args.bags) * len(SEEDS)
    print(f"wrote {len(experiments)} yaml files to {yaml_dir}/ and {json_file} ({n_runs} runs)")


if __name__ == "__main__":
    main()
