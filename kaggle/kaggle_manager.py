"""Kaggle Manager CLI for 8Chess NNUE training on Kaggle GPU.

Commands:
    python -m kaggle.kaggle_manager push     # pushes notebook and starts Kaggle GPU run
    python -m kaggle.kaggle_manager status   # checks remote execution status
    python -m kaggle.kaggle_manager pull     # downloads trained .nnue artifact to networks/
"""

import argparse
import json
import os
import subprocess
import sys
import time

KERNEL_ID = "srinivaskasukurthy/eightchess-lc0-nnue-trainer"
KAGGLE_DIR = os.path.dirname(os.path.abspath(__file__))

def push():
    print(f"Pushing {KERNEL_ID} to Kaggle...")
    cmd = ["kaggle", "kernels", "push", "-p", KAGGLE_DIR]
    subprocess.check_call(cmd)
    print("\nKernel successfully queued on Kaggle GPU!")
    print(f"Check status with: python -m kaggle.kaggle_manager status")

def status():
    cmd = ["kaggle", "kernels", "status", KERNEL_ID]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        if "Permission" in res.stderr or "404" in res.stderr:
            print(f"Kernel '{KERNEL_ID}' has not been pushed yet or is not yet initialized.")
            print(f"Run 'python3 kaggle/kaggle_manager.py push' to upload and start the run.")
        else:
            print(res.stderr.strip() or res.stdout.strip())
    else:
        print(res.stdout.strip())

def pull(out_dir="networks"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading output artifacts from {KERNEL_ID} into {out_dir}/...")
    cmd = ["kaggle", "kernels", "output", KERNEL_ID, "-p", out_dir]
    subprocess.check_call(cmd)
    print(f"Downloaded contents of {out_dir}:")
    for f in os.listdir(out_dir):
        p = os.path.join(out_dir, f)
        print(f"  {f} ({os.path.getsize(p):,} bytes)")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("push")
    sub.add_parser("status")
    pull_p = sub.add_parser("pull")
    pull_p.add_argument("--out-dir", default="networks")

    args = parser.parse_args()
    if args.command == "push":
        push()
    elif args.command == "status":
        status()
    elif args.command == "pull":
        pull(args.out_dir)

if __name__ == "__main__":
    main()
