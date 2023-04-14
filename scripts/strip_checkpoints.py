#!/usr/bin/env python3
"""
Strips PL checkpoints.
"""
import argparse
import hashlib
from pathlib import Path

import torch


class CONSTANTS:
    OUT_EXT = ".pth"

    MODEL_STATE_DICT_KEY = "state_dict"

    MODEL_PREFIX = "model."
    EMA_PREFIX = "ema.module."

    TMP_OUT_FILE = "_checkpoint.tmp" + OUT_EXT


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--ckpt",
    "--checkpoint",
    dest="checkpoint",
    metavar="PATH",
    required=True,
    type=Path,
    help="The PL checkpoint to strip.",
)
parser.add_argument(
    "-o",
    "--out",
    "--output",
    dest="output",
    metavar="PATH",
    type=Path,
    help="Output path. By default CWD. Extension will automatically be added if not present.",
)
parser.add_argument(
    "--ema", default=False, action="store_true", help="Strip out EMA weights instead!"
)

args = parser.parse_args()
if not args.checkpoint.exists():
    raise FileNotFoundError(f"Nothing found at '{args.checkpoint}'")


def strip(ckpt: Path) -> None:
    # load state and remove prefix
    pl_state_dict = torch.load(ckpt, map_location="cpu")
    prefix = CONSTANTS.EMA_PREFIX if args.ema else CONSTANTS.MODEL_PREFIX
    N = len(prefix)
    state_dict = {
        key[N:]: value
        for key, value in pl_state_dict[CONSTANTS.MODEL_STATE_DICT_KEY].items()
        if key.startswith(prefix)
    }
    epoch = pl_state_dict["epoch"]
    print(f"Loaded state dict {'(EMA) ' * args.ema}from {ckpt} with {epoch=}")

    # save as tmp name and calculate hash
    tmp_out_path = Path(CONSTANTS.TMP_OUT_FILE)
    torch.save(state_dict, tmp_out_path)
    with open(tmp_out_path, "rb") as fh:
        filehash = hashlib.sha256(fh.read()).hexdigest()

    # get proper output location components
    if args.output is not None:
        root, out_name = args.output.parent, args.output.stem
        suffix = args.output.suffix or CONSTANTS.OUT_EXT
    else:
        root = Path.cwd()
        out_name = ckpt.parent.name
        suffix = CONSTANTS.OUT_EXT

    # put hash into name and rename
    out_path = Path(root, f"{out_name}-{filehash[:10]}").with_suffix(suffix)
    tmp_out_path.rename(out_path)
    print(f"Saved to '{out_path}'")
    print(f"SHA256 hash={filehash}")


strip(ckpt=args.checkpoint)
