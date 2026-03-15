#!/usr/bin/env python3
"""List files that share the same inode (hard links)."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def iter_paths(inputs: Iterable[str], recursive: bool = True) -> Iterable[Path]:
    """Yield file paths from input files/directories."""
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            if recursive:
                for p in path.rglob("*"):
                    if p.is_file() and not p.is_symlink():
                        yield p
            else:
                for p in path.iterdir():
                    if p.is_file() and not p.is_symlink():
                        yield p
        elif path.is_file() and not path.is_symlink():
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List groups of files that are hard links to the same inode."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files/directories to scan (default: current directory).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subdirectories.",
    )
    args = parser.parse_args()

    inode_groups: dict[tuple[int, int], set[str]] = defaultdict(set)
    for path in iter_paths(args.paths, recursive=not args.no_recursive):
        try:
            st = path.stat()
        except OSError:
            continue
        inode_groups[(st.st_dev, st.st_ino)].add(str(path))

    found = False
    for group in sorted(inode_groups.values(), key=lambda g: sorted(g)[0]):
        if len(group) > 1:
            found = True
            print(" ".join(sorted(group)))

    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
