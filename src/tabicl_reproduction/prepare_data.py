"""Download the team's benchmark registry once, before launching GPU jobs."""

import argparse
import json
from pathlib import Path

from tabicl_reproduction.data import TABPFN_VAL_DATASETS, OpenMLDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+")
    args = parser.parse_args()
    ids = tuple(args.ids or TABPFN_VAL_DATASETS.values())
    source = OpenMLDataset(dataset_ids=ids, data_dir=args.data_dir)
    errors = source.download_all()
    manifest = {
        "requested_ids": ids,
        "successful_ids": [i for i in ids if i not in errors],
        "errors": {str(i): str(error) for i, error in errors.items()},
    }
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps(manifest, indent=2))
    if errors:
        raise SystemExit("Some downloads failed; see download_manifest.json")


if __name__ == "__main__":
    main()
