"""Create the split files BrepMFR's data loader expects.

The loader reads <dataset_path>/train.txt, val.txt and test.txt (stage 1), or
s_train/s_val/s_test.txt in the source folder and t_train/t_val/t_test.txt in the
target folder (stage 2 domain adaptation). Each line is a .bin file name without
its extension, e.g. "cadsynth_00123".

Examples
  # Stage 1, paper's CADSynth split (80/10/10):
  python tools/make_splits.py /data/CADSynth/bin --ratios 0.8 0.1 0.1

  # Small smoke-test subset: 2,000 models in total
  python tools/make_splits.py /data/CADSynth/bin --limit 2000

  # Stage 2: write s_*.txt for the source folder and t_*.txt for the target folder
  python tools/make_splits.py /data/MFCADpp/bin --prefix s_ --ratios 0.7 0.15 0.15
  python tools/make_splits.py /data/CADSynth/bin --prefix t_

Only files whose name ends in _<integer>.bin are used, matching the loader's rule.
Existing split files are never overwritten unless --force is given.
"""
import argparse
import random
import re
from pathlib import Path

PATTERN = re.compile(r".*_\d+$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_path", type=Path, help="folder containing the .bin graphs (searched recursively)")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--prefix", default="", help='"" for stage 1, "s_" for source, "t_" for target')
    ap.add_argument("--limit", type=int, default=None, help="use only this many models (random sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        ap.error("--ratios must sum to 1")

    stems = sorted({p.stem for p in args.dataset_path.rglob("*.bin") if PATTERN.match(p.stem)})
    if not stems:
        ap.error(f"no *_<int>.bin files found under {args.dataset_path}")

    rng = random.Random(args.seed)
    rng.shuffle(stems)
    if args.limit:
        stems = stems[: args.limit]

    n = len(stems)
    n_train = int(round(n * args.ratios[0]))
    n_val = int(round(n * args.ratios[1]))
    splits = {
        "train": stems[:n_train],
        "val": stems[n_train : n_train + n_val],
        "test": stems[n_train + n_val :],
    }

    for name, items in splits.items():
        out = args.dataset_path / f"{args.prefix}{name}.txt"
        if out.exists() and not args.force:
            print(f"skip   {out} (exists; use --force to overwrite)")
            continue
        out.write_text("\n".join(items) + "\n", encoding="utf-8")
        print(f"wrote  {out}  ({len(items)} models)")


if __name__ == "__main__":
    main()
