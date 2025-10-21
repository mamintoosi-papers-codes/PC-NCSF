"""Orchestrate paired runs of the unconditional and conditional training scripts.

This script shells out to the existing training scripts so we don't duplicate
the long training code. It ensures both scripts write into the same run
folder by supplying the same --tag and --dataset. Optionally it can skip one
or both trainings and only call the report generator.

Usage examples (Windows cmd):
  python run_pair.py --dataset scop_easy --bs 256 --epochs 40 --hd 128 --ed 8 --tag exp01
  python run_pair.py --report-only --tag exp01 --dataset scop_easy

Behavior:
 - Runs unconditional script (`circularspline_protein.py`) and conditional script
   (`circularspline_protein_embedding_c.py`) sequentially unless skipped.
 - Passes the same --tag so both store outputs under runs/<dataset>/<tag>/
 - After runs, optionally invokes report_generator.py with --runs-dir pointing
   to the run folder (so reports are generated without rescanning unrelated runs).
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import os
from typing import List
import shutil


def _build_cmd(script: str, args: List[str]) -> List[str]:
    # Use the current python executable
    return [sys.executable, script] + args


def _call(cmd: List[str]) -> int:
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="Run paired unconditional+conditional experiments and generate reports.")
    parser.add_argument("--dataset", type=str, default="torus_protein")
    parser.add_argument("--bs", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hd", type=int, default=128, help="hidden dim")
    parser.add_argument("--ed", type=int, default=32, help="embedding dim for conditional")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--tag", type=str, default=None, help="If provided, used as run folder name to group both runs")
    parser.add_argument("--save-dir", type=str, default="runs")
    parser.add_argument("--skip-uncond", action="store_true")
    parser.add_argument("--skip-cond", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Do not run training, only generate reports for the run folder")
    parser.add_argument("--report-generator-args", type=str, default="", help="Extra args to pass to report_generator.py")
    args = parser.parse_args()

    repo_root = os.path.dirname(__file__)
    uncond_script = os.path.join(repo_root, "circularspline_protein.py")
    cond_script = os.path.join(repo_root, "circularspline_protein_embedding_c.py")
    report_script = os.path.join(repo_root, "report_generator.py")

    # Determine run folder path
    if args.tag:
        run_folder = os.path.join(args.save_dir, args.dataset, args.tag)
    else:
        # fall back to conventional name used by scripts
        run_name_uncond = f"uncond_bs{args.bs}_ep{args.epochs}_hd{args.hd}"
        run_folder = os.path.join(args.save_dir, args.dataset, run_name_uncond)

    os.makedirs(run_folder, exist_ok=True)

    if args.report_only:
        print("Skipping training, only generating reports for:", run_folder)
        cmd = [sys.executable, report_script, "--runs-dir", run_folder]
        if args.dataset:
            cmd += ["--dataset", args.dataset]
        if args.report_generator_args:
            # Allow passing additional args as a space-separated string
            cmd += args.report_generator_args.split()
        return_code = _call(cmd)
        sys.exit(return_code)

    # Build common args for both scripts
    common = ["--dataset", args.dataset, "--batch-size", str(args.bs), "--epochs", str(args.epochs), "--hidden-dim", str(args.hd), "--save-dir", args.save_dir]
    if args.tag:
        common += ["--tag", args.tag]
    if args.lr is not None:
        # pass learning rate to both if set
        common += ["--lr", str(args.lr)]

    # Run unconditional
    if not args.skip_uncond:
        cmd_uncond = _build_cmd(uncond_script, common)
        rc = _call(cmd_uncond)
        if rc != 0:
            print("Unconditional script failed with code", rc)
            sys.exit(rc)

        # Move unconditional artifacts into a subfolder so conditional run doesn't overwrite them
        try:
            uncond_sub = os.path.join(run_folder, "uncond")
            os.makedirs(uncond_sub, exist_ok=True)
            for name in os.listdir(run_folder):
                src = os.path.join(run_folder, name)
                dst = os.path.join(uncond_sub, name)
                # skip the newly-created uncond folder itself
                if os.path.abspath(src) == os.path.abspath(uncond_sub):
                    continue
                # only move files (not directories) to keep any pre-existing folders
                if os.path.isfile(src):
                    try:
                        shutil.move(src, dst)
                    except Exception:
                        # non-fatal: print and continue
                        print(f"Warning: couldn't move {src} to {dst}")
        except Exception as ex:
            print("Warning: failed to snapshot unconditional artifacts:", ex)

    # Run conditional
    if not args.skip_cond:
        cond_args = common + ["--embedding-dim", str(args.ed)]
        # pass --plot-all to conditional by default to keep original behavior minimal; user can modify script if desired
        cmd_cond = _build_cmd(cond_script, cond_args + ["--plot-all"]) if True else _build_cmd(cond_script, cond_args)
        rc = _call(cmd_cond)
        if rc != 0:
            print("Conditional script failed with code", rc)
            sys.exit(rc)

    # After successful runs, generate reports for that run folder
    cmd_report = [sys.executable, report_script, "--runs-dir", run_folder]
    if args.dataset:
        cmd_report += ["--dataset", args.dataset]
    if args.report_generator_args:
        cmd_report += args.report_generator_args.split()

    rc = _call(cmd_report)
    if rc != 0:
        print("Report generation failed with code", rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
