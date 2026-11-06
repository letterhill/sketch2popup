"""사용법: python scripts/run_experiment.py --config configs/exp_D_full.yaml --split dev"""
import argparse, subprocess, yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"[{cfg['name']}] split={args.split} commit={commit}")
    # TODO(P1): splits csv 로드 → pipeline.run → iou/boundary_f1/time → results/<name>_<split>.csv


if __name__ == "__main__":
    main()
