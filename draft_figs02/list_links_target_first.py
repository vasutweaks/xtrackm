#!/usr/bin/env python3
"""List files with symlink targets shown first."""

import argparse
import glob
import os


def list_links_target_first(patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    for path in sorted(set(files)):
        if os.path.islink(path):
            # print(f"{os.readlink(path)} <- {path}")
            print(f'"{os.readlink(path)}":"{path}",')
        else:
            print(path)


def main():
    parser = argparse.ArgumentParser(
        description="Print symlinks as 'target <- link'."
    )
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["*"],
        help="Glob pattern(s), e.g. 'fig??.py' or 'fig_???.py'",
    )
    args = parser.parse_args()
    list_links_target_first(args.patterns)


if __name__ == "__main__":
    main()
